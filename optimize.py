import os
import re
import sys
import json
import dspy
import shutil
import argparse
from utils import *
from logging_utils import *
from datetime import datetime
from typing import List, Dict, Any

def create_dynamic_signature(task_config: Dict[str, Any]):
    """
    Create a dynamic DSPy Signature based on task configuration.
    
    Args:
        task_config: Task configuration dictionary containing 'data_config' and 'base_instruction'
        
    Returns:
        A dynamically created DSPy Signature class
    """
    # Get base instruction
    base_instruction = task_config.get('base_instruction', 'Process the input data and generate the output.')
    
    # Prepare field definitions
    fields = {}
    input_fields = []
    output_fields = []
    
    for field_config in task_config['data_config']:
        field_name = field_config['name']
        field_type = field_config['type']
        field_desc = field_config.get('description', f'{field_name} field')
        
        if field_type == 'input':
            fields[field_name] = (str, dspy.InputField(desc=field_desc))
            input_fields.append(field_name)
        elif field_type == 'output':
            fields[field_name] = (str, dspy.OutputField(desc=field_desc))
            output_fields.append(field_name)
        else:
            raise ValueError(f"Unknown field type: {field_type}. Must be 'input' or 'output'")
    
    # Create the signature class dynamically
    signature_class = type(
        'DynamicTaskSignature',
        (dspy.Signature,),
        {
            '__doc__': base_instruction,
            '__annotations__': {name: typ for name, (typ, _) in fields.items()},
            **{name: field for name, (_, field) in fields.items()}
        }
    )
    
    return signature_class, input_fields, output_fields


def create_lm(model_name, api_base, api_keys, max_tokens=4096):
    """Create a separate LM instance"""
    rotating_key = RotatingAPIKey(api_keys)
    generator_lm = dspy.LM(
        model=model_name,
        api_key=rotating_key,
        api_base=api_base,
        max_tokens=max_tokens,
        num_retries=10,
    )
    return generator_lm


class DSPyGenerator(dspy.Module):
    """Generator module for task execution"""
    def __init__(self, generator_lm, task_signature):
        super().__init__()
        # Store the generator LM for use in forward()
        self.generator_lm = generator_lm
        # Initialize predict with the default LM, but we'll override in forward()
        self.predict = dspy.ChainOfThought(task_signature)
    
    def forward(self, **kwargs):
        # Ensure we use the generator LM for inference
        with dspy.context(lm=self.generator_lm):
            result = self.predict(**kwargs, provide_traceback=True)
        return result


class DSPyPipeline(dspy.Module):
    """Simplified DSPy pipeline that can be directly optimized"""
    
    def __init__(self, generator_lm, task_signature):
        super().__init__()
        self.generator = DSPyGenerator(generator_lm=generator_lm, task_signature=task_signature)
    
    def forward(self, **kwargs):
        """Forward pass that generates output directly"""
        gen_result = self.generator(**kwargs)
        return gen_result


class DSPyTrainer():
    def __init__(self, args, task_config):
        self.args = args
        self.task_config = task_config
        
        # Configure DSPy with optimizer model
        api_keys = get_sambanova_api_keys()
        api_base = "https://api.sambanova.ai/v1"

        assert len(api_keys) > 0, "Need to provide at least one api_key"

        print(f"Configuring DSPy with optimizer model: {args.optimizer_model}")
        self.default_lm = create_lm(args.optimizer_model, api_base, api_keys, max_tokens=4096)
        dspy.configure(lm=self.default_lm)

        print(f"Configuring generator model: {args.generator_model}")
        self.generator_lm = create_lm(args.generator_model, api_base, api_keys, max_tokens=args.generator_max_tokens)
        self.reflection_lm = create_lm(args.reflection_model, api_base, api_keys, max_tokens=4096)

        # Create dynamic signature based on task config
        self.task_signature, self.input_fields, self.output_fields = create_dynamic_signature(task_config)
        print(f"Created dynamic signature with input fields: {self.input_fields}, output fields: {self.output_fields}")
        
        # Initialize data processor with eval metric from config
        eval_metric = task_config.get('eval_metric', 'exact_match')
        self.processor = DataProcessor(eval_metric=eval_metric)
        
        # Prepare train/val/test data
        train_samples = process_data(task_config["train_data"])
    
        if args.num_train_samples != -1:
            train_samples = train_samples[:args.num_train_samples]

        if task_config.get("train_data", "") != task_config.get("val_data", "") and task_config.get("val_data", ""):
            val_samples = process_data(task_config["val_data"])
        else:
            # If train and val uses the same file, validation set is fetched after the number of train samples
            assert args.num_train_samples != -1, "When using the same file for train and val, num_train_samples must be specified"
            assert args.num_train_samples + args.num_val_samples <= len(train_samples), f"The sum of num_train_samples ({args.num_train_samples}) and num_val_samples ({args.num_val_samples}) should be no more than the total number of samples ({len(train_samples)})."
            val_samples = train_samples[args.num_train_samples:]

        if args.num_val_samples != -1:
            val_samples = val_samples[:args.num_val_samples]

        test_samples = process_data(task_config["test_data"])
        print(f"Training on {len(train_samples)} examples, validating on {len(val_samples)}, testing on {len(test_samples)}")
        
        self.train_samples = self.convert_dataset_to_dspy_format(train_samples)
        self.val_samples = self.convert_dataset_to_dspy_format(val_samples)
        self.test_samples = self.convert_dataset_to_dspy_format(test_samples)
        
        self.task_name = task_config['task_name']
        
        # Create pipeline with the dynamic signature
        self.pipeline = DSPyPipeline(generator_lm=self.generator_lm, task_signature=self.task_signature)
        
        print("Model configuration:")
        print(f"  Default DSPy LM: {dspy.settings.lm.model}")
        print(f"  Generator LM: {self.generator_lm.model}")

    def convert_dataset_to_dspy_format(self, dataset):
        """
        Convert dataset to DSPy format dynamically based on task configuration.
        
        Args:
            dataset: List of dictionaries containing the raw data
            
        Returns:
            List of dspy.Example objects
        """
        dspy_dataset = []
        
        for d in dataset:
            # Create kwargs dict with all fields from the data
            example_kwargs = {}
            
            # Add all input fields
            for input_field in self.input_fields:
                if input_field not in d:
                    raise ValueError(f"Input field '{input_field}' not found in data: {d.keys()}")
                example_kwargs[input_field] = d[input_field]
            
            # Add all output fields
            for output_field in self.output_fields:
                if output_field not in d:
                    raise ValueError(f"Output field '{output_field}' not found in data: {d.keys()}")
                example_kwargs[output_field] = d[output_field]
            
            # Create the example and mark input fields
            example = dspy.Example(**example_kwargs).with_inputs(*self.input_fields)
            dspy_dataset.append(example)
            
        return dspy_dataset

    def get_accuracy(self, dataset, pipeline):
        """
        Evaluate accuracy on a dataset.
        
        Args:
            dataset: List of dspy.Example objects
            pipeline: The DSPy pipeline to evaluate
            
        Returns:
            Tuple of (accuracy, num_correct)
        """
        predicted = []
        ground_truth = []
        
        # Get the first output field as the primary output to evaluate
        # If there are multiple output fields, this evaluates the first one
        primary_output_field = self.output_fields[0]
        
        for i, example in enumerate(dataset):
            # Prepare input kwargs
            input_kwargs = {field: getattr(example, field) for field in self.input_fields}
            
            # Run the pipeline
            result = pipeline(**input_kwargs)
            
            # Extract the predicted output
            pred_value = getattr(result, primary_output_field, None)
            gt_value = getattr(example, primary_output_field, None)
            
            predicted.append(pred_value)
            ground_truth.append(gt_value)
            
            is_correct = self.processor.answer_is_correct(pred_value, gt_value)
            print(f"Sample {i+1}:")
            print(f"  Predicted: {pred_value}")
            print(f"  Ground truth: {gt_value}")
            print(f"  Correct: {is_correct}")
            print("="*80)

        return self.processor.evaluate_accuracy(predicted, ground_truth)

    def individual_question_accuracy_metric(self, example, pred, *args):
        """
        Metric function for DSPy optimization that computes individual accuracy.
        
        Args:
            example: The example being evaluated
            pred: The prediction from the model
            
        Returns:
            Float score (1.0 for correct, 0.0 for incorrect)
        """
        # Get the first output field as the primary field to evaluate
        primary_output_field = self.output_fields[0]
        
        if not hasattr(pred, primary_output_field):
            return 0.0
        
        predicted = getattr(pred, primary_output_field)
        ground_truth = getattr(example, primary_output_field)
        
        return float(self.processor.answer_is_correct(predicted, ground_truth))

    def train(self):
        """Train and optimize the DSPy pipeline"""
        
        results = []
        results.append({
            "train_size": len(self.train_samples),
            "val_size": len(self.val_samples),
            "test_size": len(self.test_samples),
        })

        results_path = os.path.join(self.args.save_path, "results.json")
        
        # Initial evaluation on test set
        print("Evaluating initial pipeline...")
        initial_accuracy, initial_correct = self.get_accuracy(self.test_samples, self.pipeline)
        results.append({
            "initial_test_accuracy": initial_accuracy,
            "initial_test_correct": initial_correct
        })
        print(f"Initial test accuracy: {initial_accuracy:.4f} ({initial_correct}/{len(self.test_samples)})")
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Configure optimizer
        print(f"Configuring GEPA optimizer with auto mode: {self.args.auto}")
        
        optimizer = dspy.GEPA(
            metric=self.individual_question_accuracy_metric,
            auto=self.args.auto,
            reflection_lm=self.reflection_lm,
            log_dir=self.args.save_path
        )
        
        pipeline_filename = "gepa_optimized_pipeline.json"
        print(f"Starting optimization...")

        optimized_pipeline = optimizer.compile(
            self.pipeline,
            trainset=self.train_samples,
            valset=self.val_samples
        )
        print(f"Optimization complete!")

        # Final evaluation on test set
        print("Evaluating optimized pipeline...")
        final_accuracy, final_correct = self.get_accuracy(self.test_samples, optimized_pipeline)
        results.append({
            "final_test_accuracy": final_accuracy,
            "final_test_correct": final_correct,
            "improvement": final_accuracy - initial_accuracy
        })
        print(f"Final test accuracy: {final_accuracy:.4f} ({final_correct}/{len(self.test_samples)})")
        print(f"Improvement: {final_accuracy - initial_accuracy:.4f}")
        
        # Save optimized pipeline
        optimized_pipeline.save(f"{self.args.save_path}/{pipeline_filename}")
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return optimized_pipeline


def main():
    parser = argparse.ArgumentParser(description='DSPy optimization system')
    parser.add_argument("--task_config_path", type=str, default="./sample_data/sample_config.json")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--num_train_samples", default=-1, type=int)
    parser.add_argument("--num_val_samples", default=-1, type=int)

    # DSPy program arguments
    parser.add_argument("--generator_model", type=str, default="sambanova/DeepSeek-V3.1", help="Model to use for the generator lm")
    parser.add_argument("--optimizer_model", type=str, default="sambanova/DeepSeek-V3.1", help="Model to use for DSPy optimization")
    parser.add_argument("--reflection_model", type=str, default="sambanova/DeepSeek-V3.1", help="Model to use for GEPA reflection lm")
    parser.add_argument("--generator_max_tokens", type=int, default=4096, help="generator model's max tokens to generate")
    
    # Optimizer hyperparameter
    parser.add_argument("--auto", type=str, default="heavy", help="Auto mode for optimizer")
    
    args = parser.parse_args()

    # Load task configuration
    with open(args.task_config_path, 'r') as f:
        config_data = json.load(f)
    
    task_config = config_data["task_config"]
    
    # Merge paths from config file into task_config if they exist at root level
    if "train_data" in config_data:
        task_config["train_data"] = config_data["train_data"]
    if "val_data" in config_data:
        task_config["val_data"] = config_data["val_data"]
    if "test_data" in config_data:
        task_config["test_data"] = config_data["test_data"]
    
    # For training mode, use the original folder naming
    folder_name = f"{task_config['task_name']}_gepa_{args.auto}"
    args.save_path = os.path.join(args.save_path, folder_name)
    
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path, exist_ok=False)

    # Setup logging after save_path is finalized
    stdout_tee, stderr_tee, original_stdout, original_stderr = setup_logging(args.save_path)
    
    try:
        print("="*80)
        print(f"STARTING DSPY TRAINING SESSION - {datetime.now()}")
        print("="*80)
        print(f"Arguments: {vars(args)}")
        print(f"Task config: {task_config}")
        print("="*80)
        
        trainer = DSPyTrainer(args, task_config)
        trainer.train()
        
        print("="*80)
        print(f"Training completed successfully!")
        print(f"Results saved to {args.save_path}")
        print("="*80)
    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        raise
    finally:
        cleanup_logging(stdout_tee, stderr_tee, original_stdout, original_stderr)


if __name__ == "__main__":
    main()