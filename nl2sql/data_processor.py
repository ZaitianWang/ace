import os
import json
from utils import extract_answer
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .sql_eval import BIRDSQLExecutor

def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load and process data from a JSONL file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data

class DataProcessor:
    """
    Processor for handling data preprocessing and evaluation.
    
    You only need to implement 3 methods:
    1. process_task_data() - Convert raw data to standardized format
    2. answer_is_correct() - Check if a prediction matches ground truth
    3. evaluate_accuracy() - Calculate overall accuracy
    
    The evaluation orchestration is handled by utils.evaluate_test_set().
    """
    
    def __init__(self, task_name: str):
        """Initialize with task name."""
        self.task_name = task_name
        self.bird_executor = BIRDSQLExecutor("nl2sql/bird_raw_data")
    
    def _prepare_input(self, item: dict) -> Tuple[str, str, str, str]:
        """
        Extract and parse data fields into (context, question, target).
        Customize this helper method for your task's data format.
        """
        db_id = item.get('db_id', '')
        context = item.get('context', '')
        question = item.get('question', '')
        target = item.get('target', '')
        return db_id, context, question, target
    
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw data into standardized format.
        
        Args:
            raw_data: Raw data loaded from JSONL
            
        Returns:
            List of dicts with keys: 'context', 'question', 'target'
        """
        processed_data = []
        
        for item in raw_data:
            # Apply any task-specific preprocess here
            db_id, context, question, target = self._prepare_input(item)
            
            processed_item = {
                "db_id": db_id,
                "context": context,      # Background information
                "question": question,    # The actual question/instruction
                "target": target,        # Ground truth answer
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        Implement task-specific comparison logic.
        
        This is called by the evaluation utilities in utils.py.
        """

        return self.bird_executor(predicted, ground_truth)
    
        # Example: exact match (case-insensitive)
        # return predicted.strip().lower() == ground_truth.strip().lower()
        
        # Or numeric comparison:
        # try:
        #     return float(predicted) == float(ground_truth)
        # except:
        #     return predicted == ground_truth
    
    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate accuracy across multiple predictions.
        
        This is called by the evaluation utilities in utils.py.
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth answers
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        correct = sum(
            1 for pred, truth in zip(predictions, ground_truths)
            if self.answer_is_correct(pred, truth)
        )
        
        return correct / len(predictions) if predictions else 0.0
    

import sqlite3
from typing import List, Tuple, Any, Optional, Union
import json

class SQLResultComparator:
    """执行SQL并比较结果的工具类"""
    
    def __init__(self, db_path: str):
        """
        初始化比较器
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
    
    def execute_sql(self, sql: str) -> Tuple[List[Tuple], str]:
        """
        在数据库中执行SQL语句
        
        Args:
            sql: SQL查询语句
            
        Returns:
            (结果列表, 错误信息) 元组，成功时错误信息为None
        """
        if not sql or not sql.strip():
            return [], "Empty SQL query"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 清理SQL（移除末尾分号）
            clean_sql = sql.strip().rstrip(';')
            
            cursor.execute(clean_sql)
            result = cursor.fetchall()
            
            conn.close()
            return result, ""
            
        except sqlite3.Error as e:
            return [], f"SQL Error: {str(e)}"
        except Exception as e:
            return [], f"Execution Error: {str(e)}"
    
    def normalize_result(self, result: List[Tuple]) -> List[Tuple]:
        """
        规范化查询结果（排序以便比较）
        
        Args:
            result: 原始查询结果
            
        Returns:
            规范化后的结果
        """
        if not result:
            return []
        
        # 尝试转换为可排序的格式
        try:
            # 如果是可哈希的类型，使用集合（但集合会丢失顺序）
            return sorted(result)
        except TypeError:
            # 对于包含不可哈希类型的行，尝试逐元素转换
            normalized = []
            for row in result:
                normalized_row = []
                for cell in row:
                    if isinstance(cell, (int, float, str, bool)):
                        normalized_row.append(cell)
                    else:
                        # 转换为字符串比较
                        normalized_row.append(str(cell))
                normalized.append(tuple(normalized_row))
            return sorted(normalized)
    
    def compare_results(self, result1: List[Tuple], result2: List[Tuple]) -> bool:
        """
        比较两个查询结果是否相同
        
        Args:
            result1: 第一个结果
            result2: 第二个结果
            
        Returns:
            True if 结果相同, False otherwise
        """
        # 处理None结果
        if result1 is None or result2 is None:
            return False
        
        # 规范化后比较
        norm1 = self.normalize_result(result1)
        norm2 = self.normalize_result(result2)
        
        return norm1 == norm2
    
    def sql_result_is_correct(self, predicted_sql: str, gold_sql: str) -> Tuple[bool, dict]:
        """
        比较预测SQL和金标准SQL的执行结果是否一致
        
        Args:
            predicted_sql: 预测的SQL语句
            gold_sql: 金标准SQL语句
            
        Returns:
            (是否一致, 详细信息字典)
        """
        # 执行金标准SQL
        gold_result, gold_error = self.execute_sql(gold_sql)
        if gold_error:
            return False, {
                "error": f"Gold SQL执行失败: {gold_error}",
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql
            }
        
        # 执行预测SQL
        pred_result, pred_error = self.execute_sql(predicted_sql)
        if pred_error:
            return False, {
                "error": f"Predicted SQL执行失败: {pred_error}",
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql,
                "gold_result": gold_result
            }
        
        # 比较结果
        is_correct = self.compare_results(gold_result, pred_result)
        
        details = {
            "is_correct": is_correct,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "gold_result_count": len(gold_result) if gold_result else 0,
            "pred_result_count": len(pred_result) if pred_result else 0,
            "gold_error": gold_error,
            "pred_error": pred_error
        }
        
        return is_correct, details


# 在现有框架中集成使用示例
class BIRDEvaluator:
    """集成到你的现有评估框架中"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db_info = self._load_db_info()
    
    def _load_db_info(self) -> dict:
        """加载数据库信息"""
        # 加载dev.json获取问题到数据库的映射
        with open(f"{self.data_dir}/dev.json", "r") as f:
            dev_data = json.load(f)
        
        # 创建映射：question_id -> db_id
        return {item["question_id"]: item["db_id"] for item in dev_data}
    
    def get_db_path(self, question_id: str) -> str:
        """获取问题对应的数据库路径"""
        db_id = self.db_info.get(question_id)
        if not db_id:
            raise ValueError(f"No database found for question_id: {question_id}")
        
        # 假设数据库文件在 databases/ 目录下
        return f"{self.data_dir}/databases/{db_id}.sqlite"
    
    def answer_is_correct(self, question_id: str, predicted: str, ground_truth: str) -> bool:
        """
        检查预测SQL与金标准SQL的执行结果是否一致
        
        Args:
            question_id: 问题ID
            predicted: 预测的SQL
            ground_truth: 金标准SQL
            
        Returns:
            True if 执行结果一致
        """
        # 获取数据库路径
        db_path = self.get_db_path(question_id)
        
        # 创建比较器并执行比较
        comparator = SQLResultComparator(db_path)
        is_correct, details = comparator.sql_result_is_correct(predicted, ground_truth)
        
        return is_correct


# 最简化的使用方式（如果你只需要核心功能）
def compare_sql_execution(db_path: str, sql1: str, sql2: str) -> bool:
    """
    最简单的接口：比较两个SQL在同一个数据库中的执行结果
    
    Args:
        db_path: 数据库文件路径
        sql1: 第一个SQL
        sql2: 第二个SQL
        
    Returns:
        执行结果是否相同
    """
    comparator = SQLResultComparator(db_path)
    is_correct, _ = comparator.sql_result_is_correct(sql1, sql2)
    return is_correct


# 使用示例
if __name__ == "__main__":
    # 示例1：直接使用
    db_path = "path/to/database.sqlite"
    
    sql1 = "SELECT name, salary FROM employees WHERE department = 'Sales'"
    sql2 = "SELECT name, salary FROM employees WHERE dept = 'Sales'"  # 假设表结构不同
    
    result = compare_sql_execution(db_path, sql1, sql2)
    print(f"SQL执行结果相同: {result}")
    
    # 示例2：集成到现有框架
    evaluator = BIRDEvaluator("path/to/bird_dataset")
    
    # 假设你知道question_id
    question_id = "question_123"
    predicted_sql = "SELECT COUNT(*) FROM orders"
    gold_sql = "SELECT COUNT(id) FROM orders"
    
    is_correct = evaluator.answer_is_correct(question_id, predicted_sql, gold_sql)
    print(f"答案是否正确: {is_correct}")