
import re

def check_template_balance(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    stack = []
    
    # Regex for tags
    tag_re = re.compile(r'{%\s*(for|if|block|with|empty|endfor|endif|endblock|endwith)\b.*?%}')

    for i, line in enumerate(lines):
        # We need to find all tags in line in order
        pos = 0
        while True:
            match = tag_re.search(line, pos)
            if not match:
                break
            
            tag_content = match.group(0)
            tag_name = match.group(1)
            
            # Start tags pushed to stack
            if tag_name in ['for', 'if', 'block', 'with']:
                stack.append({'tag': tag_name, 'line': i + 1, 'content': tag_content})
            
            # End tags pop from stack
            elif tag_name in ['endfor', 'endif', 'endblock', 'endwith']:
                if not stack:
                    print(f"Error: Unexpected {{% {tag_name} %}} at line {i+1}. Stack is empty.")
                    return

                last = stack[-1]
                expected_end = 'end' + last['tag']
                
                if tag_name == expected_end:
                    stack.pop()
                else:
                    print(f"Error: Mismatch at line {i+1}. Found {{% {tag_name} %}}, expected {{% {expected_end} %}} for {{% {last['tag']} %}} opened at line {last['line']}.")
                    return
            
            pos = match.end()

    if stack:
        print("Error: Unclosed tags remaining at end of file:")
        for item in stack:
            print(f"  Unclosed {{% {item['tag']} %}} opened at line {item['line']}: {item['content']}")
    else:
        print("Success: All tags are balanced.")

if __name__ == "__main__":
    check_template_balance("analysis/templates/analysis/analysis_result.html")
