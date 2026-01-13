import sqlite3
import os
from typing import Dict, List, Tuple, Optional, Any

class BIRDSQLExecutor:
    """
    自动从dev_gold.sql查找数据库并执行SQL比较的类
    """
    
    def __init__(self, dataset_path: str):
        """
        初始化类
        
        Args:
            dataset_path: BIRD数据集路径，包含dev_gold.sql和databases目录
        """
        self.dataset_path = dataset_path
        self.gold_to_db_map = self._load_gold_sql_mapping()
    
    def _load_gold_sql_mapping(self) -> Dict[str, str]:
        """
        从dev_gold.sql加载SQL到数据库的映射
        
        文件格式示例:
        SELECT time FROM lapTimes ORDER BY time LIMIT 1    formula_1
        SELECT * FROM employees WHERE department = 'Sales'    hr_db
        
        返回: {sql: db_name} 映射
        """
        mapping = {}
        gold_file = os.path.join(self.dataset_path, "dev_gold.sql")
        
        if not os.path.exists(gold_file):
            raise FileNotFoundError(f"dev_gold.sql not found at {gold_file}")
        
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 查找最后一个空格或制表符分隔数据库名
                # 假设格式: SQL<空格或制表符>db_name
                # 我们需要找到最后一个分隔位置
                last_space = line.rfind(' ')
                last_tab = line.rfind('\t')
                separator_pos = max(last_space, last_tab)
                
                if separator_pos > 0:
                    sql = line[:separator_pos].strip()
                    db_name = line[separator_pos:].strip()
                    
                    # 如果SQL以分号结尾，去掉分号
                    if sql.endswith(';'):
                        sql = sql[:-1]
                    
                    # 只记录第一次出现的映射
                    if sql not in mapping:
                        mapping[sql] = db_name
        
        print(f"Loaded {len(mapping)} SQL-to-DB mappings from dev_gold.sql")
        return mapping
    
    def _find_db_for_sql(self, sql_query: str) -> Optional[str]:
        """
        为给定的SQL查询查找对应的数据库名
        
        Args:
            sql_query: SQL查询语句
            
        Returns:
            数据库名，如果找不到返回None
        """
        # 清理SQL（去掉分号，去除两端空格）
        clean_sql = sql_query.strip()
        if clean_sql.endswith(';'):
            clean_sql = clean_sql[:-1]
        
        # 尝试精确匹配
        if clean_sql in self.gold_to_db_map:
            return self.gold_to_db_map[clean_sql]
        
        # 如果精确匹配失败，尝试近似匹配（去除多余空格）
        normalized_sql = ' '.join(clean_sql.split())
        if normalized_sql in self.gold_to_db_map:
            return self.gold_to_db_map[normalized_sql]
        
        # 遍历所有键，寻找相似的SQL
        for gold_sql in self.gold_to_db_map:
            # 简单的相似性检查：相同的单词集合（忽略顺序）
            gold_words = set(gold_sql.lower().split())
            query_words = set(clean_sql.lower().split())
            
            # 如果80%以上的单词匹配，认为是相同的查询
            if len(gold_words & query_words) / max(len(gold_words), len(query_words)) > 0.8:
                return self.gold_to_db_map[gold_sql]
        
        return None
    
    def _get_db_path(self, db_name: str) -> str:
        """
        根据数据库名获取数据库文件路径
        
        Args:
            db_name: 数据库名
            
        Returns:
            数据库文件路径
        """
        # 尝试常见的扩展名
        possible_paths = [
            os.path.join(self.dataset_path, "databases/dev_databases", f"{db_name}/{db_name}.sqlite"),
            os.path.join(self.dataset_path, "databases/dev_databases", f"{db_name}/{db_name}.db"),
            os.path.join(self.dataset_path, f"{db_name}.sqlite"),
            os.path.join(self.dataset_path, f"{db_name}.db"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(f"Database {db_name} not found in any expected location")
    
    def execute_query(self, db_path: str, sql: str) -> Tuple[List[Tuple], str]:
        """
        在指定数据库中执行SQL查询
        
        Args:
            db_path: 数据库文件路径
            sql: SQL查询语句
            
        Returns:
            (结果列表, 错误信息) 元组
        """
        if not sql or not sql.strip():
            return [], "Empty SQL query"
        
        try:
            conn = sqlite3.connect(db_path)
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
    
    def _normalize_result(self, result: List[Tuple]) -> List[Tuple]:
        """
        规范化查询结果以便比较
        
        Args:
            result: 原始查询结果
            
        Returns:
            规范化后的结果
        """
        if not result:
            return []
        
        # 尝试排序比较
        try:
            # 转换为可排序的格式
            normalized = []
            for row in result:
                norm_row = []
                for cell in row:
                    # 处理不同类型的值
                    if cell is None:
                        norm_row.append(None)
                    elif isinstance(cell, (int, float, str, bool)):
                        norm_row.append(cell)
                    else:
                        # 其他类型转换为字符串
                        norm_row.append(str(cell))
                normalized.append(tuple(norm_row))
            return sorted(normalized)
        except:
            # 如果排序失败，返回原始结果
            return result
    
    def compare_results(self, result1: List[Tuple], result2: List[Tuple]) -> bool:
        """
        比较两个查询结果是否相同
        
        Args:
            result1: 第一个结果
            result2: 第二个结果
            
        Returns:
            True if 结果相同, False otherwise
        """
        if result1 is None or result2 is None:
            return False
        
        # 规范化后比较
        norm1 = self._normalize_result(result1)
        norm2 = self._normalize_result(result2)
        
        return norm1 == norm2
    
    def __call__(self, predicted: str, ground_truth: str) -> bool:
        """
        主调用方法：比较预测SQL和真实SQL的执行结果
        
        Args:
            predicted: 预测的SQL语句
            ground_truth: 真实的SQL语句（金标准）
            
        Returns:
            True if 执行结果相同, False otherwise
        """
        # 1. 根据真实SQL查找数据库名
        db_name = self._find_db_for_sql(ground_truth)
        if not db_name:
            print(f"Warning: Could not find database for SQL: {ground_truth[:50]}...")
            # 可以退回到字符串比较
            return predicted.strip().lower() == ground_truth.strip().lower()
        
        # 2. 获取数据库路径
        try:
            db_path = self._get_db_path(db_name)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            # 退回到字符串比较
            return predicted.strip().lower() == ground_truth.strip().lower()
        
        # 3. 执行真实SQL
        gold_result, gold_error = self.execute_query(db_path, ground_truth)
        if gold_error:
            print(f"Error executing gold SQL: {gold_error}")
            # 退回到字符串比较
            return predicted.strip().lower() == ground_truth.strip().lower()
        
        # 4. 执行预测SQL
        pred_result, pred_error = self.execute_query(db_path, predicted)
        if pred_error:
            print(f"Error executing predicted SQL: {pred_error}")
            return False
        
        # 5. 比较结果
        return self.compare_results(gold_result, pred_result)


# 在你的现有框架中使用
class YourEvaluator:
    def __init__(self, dataset_path: str):
        self.bird_executor = BIRDSQLExecutor(dataset_path)
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        
        This is called by the evaluation utilities in utils.py.
        """
        # 使用BIRDSQLExecutor进行执行结果比较
        return self.bird_executor(predicted, ground_truth)


# 使用示例
if __name__ == "__main__":
    # 假设你的数据集路径
    dataset_path = "nl2sql/bird_raw_data"
    
    # 创建评估器
    evaluator = YourEvaluator(dataset_path)
    
    # 示例测试
    ground_truth = "SELECT duration FROM pitStops ORDER BY duration DESC LIMIT 1"
    predicted1 = "SELECT MAX(CAST(duration AS INTEGER)) FROM pitStops;"  # pre
    predicted2 = "SELECT duration FROM pitStops ORDER BY duration DESC LIMIT 1;"  # post
    predicted3 = "SELECT * FROM lapTimes"  # other
    
    print(f"Test 1 (pre): {evaluator.answer_is_correct(predicted1, ground_truth)}")
    print(f"Test 2 (post): {evaluator.answer_is_correct(predicted2, ground_truth)}")
    print(f"Test 3 (other): {evaluator.answer_is_correct(predicted3, ground_truth)}")
    
    # 也可以直接使用BIRDSQLExecutor
    executor = BIRDSQLExecutor(dataset_path)
    result = executor(predicted1, ground_truth)
    print(f"Direct test: {result}")