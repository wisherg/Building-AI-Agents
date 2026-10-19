import inspect
import typing
from typing import get_type_hints

def generate_tool_description(func) -> dict:
    """
    根据给定的工具函数，自动生成符合OpenAI规范的工具描述
    
    Args:
        func: 工具函数对象
        
    Returns:
        符合OpenAI规范的工具描述字典
    """
    # 获取函数名称
    func_name = func.__name__
    
    # 获取函数描述（从docstring提取）
    func_doc = inspect.getdoc(func) or ""
    description = func_doc.strip().split('\n')[0] if func_doc else f"调用函数 {func_name}"
    
    # 获取函数参数信息
    sig = inspect.signature(func)
    parameters = {}
    required = []
    
    # 获取类型提示
    type_hints = get_type_hints(func)
    
    for param_name, param in sig.parameters.items():
        # 跳过*args和**kwargs
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        
        # 获取参数类型
        param_type = type_hints.get(param_name, type(None))
        
        # 转换为JSON Schema类型
        json_type = _python_type_to_json_type(param_type)
        
        # 获取参数描述（从docstring提取或使用默认）
        param_desc = _extract_param_description(func_doc, param_name) or f"{param_name}参数"
        
        parameters[param_name] = {
            "type": json_type,
            "description": param_desc
        }
        
        # 判断是否为必需参数
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    # 构建工具描述
    tool_description = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required
            }
        }
    }
    
    return tool_description

def _python_type_to_json_type(py_type) -> str:
    """将Python类型转换为JSON Schema类型"""
    # 处理Optional类型
    if typing.get_origin(py_type) is typing.Union:
        args = typing.get_args(py_type)
        # 如果是Optional（包含None），取第一个非None类型
        for arg in args:
            if arg is not type(None):
                return _python_type_to_json_type(arg)
        return "string"
    
    # 基础类型映射
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null"
    }
    
    # 处理泛型类型
    origin = typing.get_origin(py_type)
    if origin is not None:
        if origin is list:
            return "array"
        elif origin is dict:
            return "object"
    
    return type_mapping.get(py_type, "string")

def _extract_param_description(docstring: str, param_name: str) -> str:
    """从docstring中提取参数描述"""
    if not docstring:
        return None
    
    # 尝试从docstring中查找参数描述
    lines = docstring.strip().split('\n')
    for line in lines:
        line = line.strip()
        # 匹配参数描述行（支持多种格式）
        if line.lower().startswith(f":param {param_name}:"):
            return line[len(f":param {param_name}:"):].strip()
        elif line.lower().startswith(f"param {param_name}:"):
            return line[len(f"param {param_name}:"):].strip()
        elif line.startswith(f"{param_name}:"):
            return line[len(f"{param_name}:"):].strip()
    
    return None

# 示例使用
if __name__ == "__main__":
    # 示例工具函数
    def get_weather(city: str, unit: str = "celsius") -> str:
        """
        获取指定城市的天气信息
        
        :param city: 城市名称
        :param unit: 温度单位，可选值：celsius（摄氏度）、fahrenheit（华氏度）
        :return: 天气信息字符串
        """
        return f"{city} 当前天气：晴天，25{chr(176)}C"
    
    def get_stock_price(symbol: str, date: str = None) -> float:
        """
        获取股票实时价格
        
        :param symbol: 股票代码，如AAPL、GOOGL
        :param date: 查询日期，格式YYYY-MM-DD，默认当前日期
        :return: 股票价格
        """
        return 175.50
    
    # 生成工具描述
    weather_tool = generate_tool_description(get_weather)
    stock_tool = generate_tool_description(get_stock_price)
    
    # 输出结果
    print("=" * 60)
    print("get_weather 工具描述：")
    print("=" * 60)
    import json
    print(json.dumps(weather_tool, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 60)
    print("get_stock_price 工具描述：")
    print("=" * 60)
    print(json.dumps(stock_tool, ensure_ascii=False, indent=2))