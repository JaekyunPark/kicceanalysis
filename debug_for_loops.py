
import re

def check_for_loops(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    stack = []
    
    # Only care about for and endfor
    tag_re = re.compile(r'{%\s*(for|endfor)\b.*?%}')

    for i, line in enumerate(lines):
        pos = 0
        while True:
            match = tag_re.search(line, pos)
            if not match:
                break
            
            tag_name = match.group(1)
            
            if tag_name == 'for':
                stack.append(i + 1)
                # print(f"PUSH for at {i+1}")
            
            elif tag_name == 'endfor':
                if not stack:
                    print(f"Error: Unexpected endfor at {i+1}")
                    return
                start_line = stack.pop()
                print(f"POP endfor at {i+1} closes for at {start_line}")
            
            pos = match.end()

    if stack:
        print(f"Unclosed loops: {stack}")

if __name__ == "__main__":
    check_for_loops("analysis/templates/analysis/analysis_result.html")
