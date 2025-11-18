import os
import sys
import logging
from datetime import datetime

class TeeOutput:
    """Class to write output to both file and console"""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w')
        self.original_stream = original_stream
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.original_stream.write(text)
        self.original_stream.flush()
        
    def flush(self):
        self.file.flush()
        self.original_stream.flush()
        
    def close(self):
        self.file.close()

def setup_logging(save_path):
    """Setup comprehensive logging to capture everything"""
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(save_path, f"training_log_{timestamp}.log")
    
    # Configure root logger to capture all logging (including from dspy)
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all levels
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)  # Also output to console
        ],
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to appropriate levels
    # You might want to adjust these based on what you want to see
    logging.getLogger('dspy').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.WARNING)  # Reduce HTTP request noise
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Create a tee for stdout and stderr to capture print statements
    stdout_tee = TeeOutput(
        os.path.join(save_path, f"stdout_{timestamp}.log"), 
        sys.__stdout__
    )
    stderr_tee = TeeOutput(
        os.path.join(save_path, f"stderr_{timestamp}.log"), 
        sys.__stderr__
    )
    
    # Store original streams for cleanup
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Redirect stdout and stderr
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    
    print(f"Logging setup complete. All output will be saved to {save_path}")
    print(f"Main log file: {log_file_path}")
    
    return stdout_tee, stderr_tee, original_stdout, original_stderr

def cleanup_logging(stdout_tee, stderr_tee, original_stdout, original_stderr):
    """Restore original streams and close log files"""
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    if stdout_tee:
        stdout_tee.close()
    if stderr_tee:
        stderr_tee.close()