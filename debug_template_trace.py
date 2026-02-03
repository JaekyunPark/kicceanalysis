
import re

def check_template_balance(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    stack = []
    
    tag_re = re.compile(r'{%\s*(for|if|block|with|empty|endfor|endif|endblock|endwith)\b.*?%}')

    for i, line in enumerate(lines):
        pos = 0
        while True:
            match = tag_re.search(line, pos)
            if not match:
                break
            
            tag_content = match.group(0)
            tag_name = match.group(1)
            
            if tag_name in ['for', 'if', 'block', 'with']:
                stack.append({'tag': tag_name, 'line': i + 1})
                print(f"[{i+1}] PUSH {tag_name} -> Stack: {[x['tag'] for x in stack]}")
            
            elif tag_name in ['endfor', 'endif', 'endblock', 'endwith']:
                if not stack:
                    print(f"Error: Unexpected {{% {tag_name} %}} at line {i+1}. Stack is empty.")
                    return

                last = stack[-1]
                expected_end = 'end' + last['tag']
                
                if tag_name == expected_end:
                    stack.pop()
                    print(f"[{i+1}] POP  {tag_name} (closed {last['tag']} from {last['line']}) -> Stack: {[x['tag'] for x in stack]}")
                else:
                    print(f"[{i+1}] Error: Mismatch. Found {{% {tag_name} %}}, expected {{% {expected_end} %}} for {{% {last['tag']} %}} opened at line {last['line']}.")
                    return
            
            pos = match.end()

if __name__ == "__main__":
    check_template_balance("analysis/templates/analysis/analysis_result.html")
