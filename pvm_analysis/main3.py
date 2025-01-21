import sys
import numpy as np
from types import FrameType
import ctypes
import inspect

class NumpyFunctionAnalyzer:
    def __init__(self, target_function):
        self.target_function = target_function
        self.call_stack = []
        self.c_calls = []
        
    def trace_function(self, frame: FrameType, event: str, arg: any) -> callable:
        if event == 'call':
            code = frame.f_code
            # Track numpy-related calls
            if 'numpy' in code.co_filename:
                call_info = {
                    'function': code.co_name,
                    'file': code.co_filename,
                    'locals': frame.f_locals.copy(),
                    'line': code.co_firstlineno
                }
                self.call_stack.append(call_info)
                
        elif event == 'return':
            if self.call_stack:
                call_info = self.call_stack[-1]
                call_info['return_value'] = arg
                
        return self.trace_function
    
    def analyze_function(self, *args, **kwargs):
        # Enable tracing
        sys.settrace(self.trace_function)
        
        # Execute the function
        result = self.target_function(*args, **kwargs)
        
        # Disable tracing
        sys.settrace(None)
        
        return self.generate_report(result)
    
    def generate_report(self, result):
        report = []
        report.append(f"Analysis of {self.target_function.__name__}:")
        report.append("\nFunction Information:")
        report.append(f"Module: {self.target_function.__module__}")
        report.append(f"Type: {type(self.target_function)}")
        
        if hasattr(self.target_function, '__doc__'):
            report.append(f"\nDocumentation:\n{self.target_function.__doc__}")
        
        report.append("\nCall Stack:")
        for idx, call in enumerate(self.call_stack, 1):
            report.append(f"\n{idx}. Function: {call['function']}")
            report.append(f"   File: {call['file']}")
            report.append(f"   Line: {call['line']}")
            report.append(f"   Local variables: {call['locals']}")
            if 'return_value' in call:
                report.append(f"   Return value: {call['return_value']}")
        
        return '\n'.join(report)

# Example usage
def analyze_zeros():
    analyzer = NumpyFunctionAnalyzer(np.zeros)
    report = analyzer.analyze_function((2, 2))
    print(report)

if __name__ == "__main__":
    analyze_zeros()
