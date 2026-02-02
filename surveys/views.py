from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import SurveyData, Codebook, RecodeRule
from .forms import SurveyDataUploadForm, CodebookUploadForm
import pandas as pd
import json
import os
from django.conf import settings
try:
    import pyreadstat
except ImportError:
    pyreadstat = None

def home(request):
    """홈페이지"""
    return render(request, 'surveys/home.html')

@login_required
def data_upload(request):
    """데이터 업로드"""
    if request.method == 'POST':
        form = SurveyDataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            survey_data = form.save(commit=False)
            survey_data.user = request.user
            survey_data.save()
            
            # .sav 파일인 경우 자동으로 코드북 생성
            if survey_data.file.name.lower().endswith('.sav'):
                try:
                    create_codebook_from_spss(survey_data)
                    messages.success(request, '데이터가 업로드되고 코드북이 자동으로 생성되었습니다.')
                except Exception as e:
                    messages.warning(request, f'데이터는 업로드되었으나 코드북 생성에 실패했습니다: {e}')
            else:
                messages.success(request, '데이터가 성공적으로 업로드되었습니다.')
                
            return redirect('surveys:data_list')
    else:
        form = SurveyDataUploadForm()
    
    return render(request, 'surveys/data_upload.html', {'form': form})

@login_required
def codebook_upload(request):
    """코드북 업로드"""
    if request.method == 'POST':
        form = CodebookUploadForm(user=request.user, data=request.POST, files=request.FILES)
        if form.is_valid():
            codebook = form.save(commit=False)
            codebook.user = request.user
            codebook.save()
            messages.success(request, '코드북이 성공적으로 업로드되었습니다.')
            return redirect('surveys:codebook_list')
    else:
        form = CodebookUploadForm(user=request.user)
    
    return render(request, 'surveys/codebook_upload.html', {'form': form})

@login_required
def data_list(request):
    """데이터 목록"""
    datasets = SurveyData.objects.filter(user=request.user)
    return render(request, 'surveys/data_list.html', {'datasets': datasets})

@login_required
def codebook_list(request):
    """코드북 목록"""
    codebooks = Codebook.objects.filter(user=request.user)
    return render(request, 'surveys/codebook_list.html', {'codebooks': codebooks})

@login_required
def data_delete(request, pk):
    """데이터 삭제"""
    dataset = get_object_or_404(SurveyData, pk=pk, user=request.user)
    if request.method == 'POST':
        dataset.delete()
        messages.success(request, '데이터가 삭제되었습니다.')
        return redirect('surveys:data_list')
    return render(request, 'surveys/data_confirm_delete.html', {'dataset': dataset})

@login_required
def codebook_delete(request, pk):
    """코드북 삭제"""
    codebook = get_object_or_404(Codebook, pk=pk, user=request.user)
    if request.method == 'POST':
        codebook.delete()
        messages.success(request, '코드북이 삭제되었습니다.')
        return redirect('surveys:codebook_list')
    return render(request, 'surveys/codebook_confirm_delete.html', {'codebook': codebook})

# ============================================
# 변수 변환 (RECODE) 관련 뷰
# ============================================

@login_required
def get_variables_for_dataset(request, dataset_id):
    """특정 데이터셋의 변수 목록 반환 (API)"""
    dataset = get_object_or_404(SurveyData, id=dataset_id, user=request.user)
    
    try:
        if dataset.file.name.lower().endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.lower().endswith('.sav'):
            df = pd.read_spss(dataset.file.path, convert_categoricals=False)
        else:
            df = pd.read_excel(dataset.file.path)
        
        variables = df.columns.tolist()
        
        return JsonResponse({
            'variables': variables,
            'count': len(variables)
        })
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

@login_required
def get_variable_values(request, dataset_id, variable):
    """특정 변수의 고유값 및 빈도 반환 (API)"""
    dataset = get_object_or_404(SurveyData, id=dataset_id, user=request.user)
    
    try:
        if dataset.file.name.lower().endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.lower().endswith('.sav'):
            df = pd.read_spss(dataset.file.path, convert_categoricals=False)
        else:
            df = pd.read_excel(dataset.file.path)
        
        if variable not in df.columns:
            return JsonResponse({'error': 'Variable not found'}, status=404)
        
        value_counts = df[variable].value_counts().sort_index()
        
        # 코드북 로드 시도
        codebook_labels = {}
        try:
            # 이 데이터셋에 연결된 코드북 찾기
            from analysis.views import load_codebook
            codebook = dataset.codebooks.first()
            if codebook:
                codebook_data = load_codebook(codebook)
                if variable in codebook_data and 'values' in codebook_data[variable]:
                    codebook_labels = codebook_data[variable]['values']
        except:
            pass
        
        values = []
        for val, count in value_counts.items():
            value_str = str(val)
            
            # 코드북 레이블 찾기
            label = None
            if codebook_labels:
                # 정수로 변환 시도
                try:
                    key = str(int(float(val)))
                except:
                    key = value_str
                label = codebook_labels.get(key)
            
            # 표시 텍스트
            if label:
                display_text = f"{value_str}: {label}"
            else:
                display_text = value_str
            
            values.append({
                'value': value_str,
                'label': label,
                'display': display_text,
                'count': int(count),
                'percentage': round(count / len(df) * 100, 1)
            })
        
        return JsonResponse({
            'values': values,
            'total': len(df),
            'has_labels': bool(codebook_labels)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e)
        }, status=500)

@login_required
def recode_variable(request):
    """변수 변환 메인 페이지"""
    datasets = SurveyData.objects.filter(user=request.user)
    existing_rules = RecodeRule.objects.filter(user=request.user).select_related('dataset')
    
    return render(request, 'surveys/recode.html', {
        'datasets': datasets,
        'existing_rules': existing_rules
    })

@login_required
def recode_preview(request):
    """변환 미리보기 (AJAX)"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    try:
        dataset_id = request.POST.get('dataset_id')
        source_variable = request.POST.get('source_variable')
        rules_json = request.POST.get('rules')
        
        dataset = get_object_or_404(SurveyData, id=dataset_id, user=request.user)
        
        # 데이터 로드
        if dataset.file.name.lower().endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.lower().endswith('.sav'):
            df = pd.read_spss(dataset.file.path, convert_categoricals=False)
        else:
            df = pd.read_excel(dataset.file.path)
        
        if source_variable not in df.columns:
            return JsonResponse({'error': 'Variable not found'}, status=404)
        
        # 규칙 파싱
        rules = json.loads(rules_json)
        
        # 변환 적용
        original_values = df[source_variable].copy()
        new_values = apply_recode_rules(original_values, rules)
        
        # 미리보기 데이터 생성
        preview_data = []
        
        # 원본 값별로 그룹화
        for orig_val in original_values.dropna().unique():
            mask = original_values == orig_val
            new_val = new_values[mask].iloc[0] if mask.any() else orig_val
            count = mask.sum()
            
            preview_data.append({
                'original': str(orig_val),
                'new': str(new_val),
                'count': int(count),
                'changed': str(orig_val) != str(new_val)
            })
        
        # 새 값별 빈도
        new_value_counts = {}
        for item in preview_data:
            new_val = item['new']
            if new_val not in new_value_counts:
                new_value_counts[new_val] = 0
            new_value_counts[new_val] += item['count']
        
        return JsonResponse({
            'success': True,
            'preview': preview_data,
            'new_value_counts': new_value_counts,
            'total': len(df)
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def recode_apply(request):
    """변환 규칙 저장 및 실제 데이터셋에 적용"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    try:
        dataset_id = request.POST.get('dataset_id')
        source_variable = request.POST.get('source_variable')
        target_variable = request.POST.get('target_variable')
        variable_label = request.POST.get('variable_label', '')  # 사용자 입력 변수 레이블
        rule_name = request.POST.get('rule_name')
        rules_json = request.POST.get('rules')
        description = request.POST.get('description', '')
        
        dataset = get_object_or_404(SurveyData, id=dataset_id, user=request.user)
        
        # 규칙 파싱
        rules = json.loads(rules_json)
        
        # 데이터 로드
        if dataset.file.name.lower().endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.lower().endswith('.sav'):
            df = pd.read_spss(dataset.file.path, convert_categoricals=False)
        else:
            df = pd.read_excel(dataset.file.path)
        
        # 원본 변수 확인
        if source_variable not in df.columns:
            return JsonResponse({'error': 'Source variable not found'}, status=404)
        
        # 변환 적용
        df[target_variable] = apply_recode_rules(df[source_variable], rules)
        
        # 파일 저장 (원본 덮어쓰기)
        if dataset.file.name.lower().endswith('.csv'):
            df.to_csv(dataset.file.path, index=False, encoding='utf-8-sig')
        elif dataset.file.name.lower().endswith('.sav'):
            # sav 파일은 덮어쓰기 대신 excel로 변환하거나 pyreadstat.write_sav 사용 가능
            # 여기서는 pyreadstat.write_sav 사용 (메타데이터 보존 어려울 수 있음)
            import pyreadstat
            pyreadstat.write_sav(df, dataset.file.path)
        else:
            df.to_excel(dataset.file.path, index=False, engine='openpyxl')
        
        # 코드북 업데이트 (연결된 코드북이 있으면)
        codebook = dataset.codebooks.first()
        if codebook:
            try:
                update_codebook_with_recode(
                    codebook, 
                    source_variable, 
                    target_variable, 
                    rules, 
                    rule_name,
                    variable_label  # 사용자 입력 레이블 전달
                )
            except Exception as e:
                print(f"코드북 업데이트 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # 규칙 저장
        recode_rule, created = RecodeRule.objects.update_or_create(
            user=request.user,
            dataset=dataset,
            target_variable=target_variable,
            defaults={
                'name': rule_name,
                'source_variable': source_variable,
                'rules': rules,
                'description': description
            }
        )
        
        message = f'변환 규칙이 저장되고 데이터셋에 "{target_variable}" 변수가 추가되었습니다.'
        if codebook:
            message += ' 코드북도 자동으로 업데이트되었습니다.'
        
        return JsonResponse({
            'success': True,
            'message': message,
            'rule_id': recode_rule.id,
            'created': created
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def recode_delete(request, rule_id):
    """변환 규칙 삭제 및 데이터셋에서 변수 제거"""
    rule = get_object_or_404(RecodeRule, id=rule_id, user=request.user)
    
    try:
        dataset = rule.dataset
        target_variable = rule.target_variable
        
        # 데이터 로드
        if dataset.file.name.lower().endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.lower().endswith('.sav'):
            df = pd.read_spss(dataset.file.path, convert_categoricals=False)
        else:
            df = pd.read_excel(dataset.file.path)
        
        # 변수가 존재하면 제거
        if target_variable in df.columns:
            df = df.drop(columns=[target_variable])
            
            # 파일 저장
            if dataset.file.name.lower().endswith('.csv'):
                df.to_csv(dataset.file.path, index=False, encoding='utf-8-sig')
            elif dataset.file.name.lower().endswith('.sav'):
                import pyreadstat
                pyreadstat.write_sav(df, dataset.file.path)
            else:
                df.to_excel(dataset.file.path, index=False, engine='openpyxl')
        
        # 코드북에서도 제거 (연결된 코드북이 있으면)
        codebook = dataset.codebooks.first()
        if codebook:
            try:
                remove_variable_from_codebook(codebook, target_variable)
            except Exception as e:
                print(f"코드북 업데이트 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # 규칙 삭제
        rule.delete()
        
        message = f'변환 규칙이 삭제되고 "{target_variable}" 변수가 데이터셋에서 제거되었습니다.'
        if codebook:
            message += ' 코드북도 함께 업데이트되었습니다.'
        
        return JsonResponse({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def apply_recode_rules(series, rules):
    """변환 규칙을 시리즈에 적용"""
    result = series.copy()
    
    for rule in rules:
        rule_type = rule.get('type')
        
        if rule_type == 'single':
            # 단일값 변환
            old_value = rule.get('old')
            new_value = rule.get('new')
            
            # 타입 변환 시도
            try:
                if series.dtype in ['int64', 'float64']:
                    old_value = float(old_value)
            except:
                pass
            
            mask = series == old_value
            result[mask] = new_value
        
        elif rule_type == 'range':
            # 범위 변환
            min_val = float(rule.get('min'))
            max_val = float(rule.get('max'))
            new_value = rule.get('new')
            
            mask = (series >= min_val) & (series <= max_val)
            result[mask] = new_value
        
        elif rule_type == 'condition':
            # 조건 변환
            operator = rule.get('operator')
            value = float(rule.get('value'))
            new_value = rule.get('new')
            
            if operator == '>=':
                mask = series >= value
            elif operator == '>':
                mask = series > value
            elif operator == '<=':
                mask = series <= value
            elif operator == '<':
                mask = series < value
            elif operator == '==':
                mask = series == value
            elif operator == '!=':
                mask = series != value
            else:
                continue
            
            result[mask] = new_value
        
        elif rule_type == 'else':
            # ELSE 처리 - 아직 변환되지 않은 값들
            new_value = rule.get('new')
            unchanged_mask = result == series
            result[unchanged_mask] = new_value
    
    return result

def update_codebook_with_recode(codebook, source_variable, target_variable, rules, rule_name, variable_label=''):
    """코드북에 변환된 변수 정보 추가 (Variable/Variable Label/Value/Value Label 구조)"""
    
    # 기존 코드북 로드
    if codebook.file.name.endswith('.csv'):
        codebook_df = pd.read_csv(codebook.file.path, encoding='utf-8-sig')
    else:
        codebook_df = pd.read_excel(codebook.file.path)
    
    print(f"=== 코드북 업데이트 시작 ===")
    print(f"기존 코드북 컬럼: {codebook_df.columns.tolist()}")
    print(f"기존 코드북 행 수: {len(codebook_df)}")
    
    # 컬럼명 매핑 (대소문자 및 공백 무시)
    col_mapping = {}
    for col in codebook_df.columns:
        col_lower = col.lower().strip()
        if 'variable' in col_lower and 'label' not in col_lower:
            col_mapping['variable'] = col
        elif 'variable' in col_lower and 'label' in col_lower:
            col_mapping['variable_label'] = col
        elif 'value' in col_lower and 'label' not in col_lower:
            col_mapping['value'] = col
        elif 'value' in col_lower and 'label' in col_lower:
            col_mapping['value_label'] = col
    
    print(f"컬럼 매핑: {col_mapping}")
    
    # 필수 컬럼 확인
    if 'variable' not in col_mapping or 'value' not in col_mapping:
        print("경고: 코드북 형식이 올바르지 않습니다.")
        return
    
    var_col = col_mapping['variable']
    val_col = col_mapping['value']
    var_label_col = col_mapping.get('variable_label', 'Variable Label')
    val_label_col = col_mapping.get('value_label', 'Value Label')
    
    # Variable Label, Value Label 컬럼이 없으면 생성
    if var_label_col not in codebook_df.columns:
        codebook_df[var_label_col] = ''
    if val_label_col not in codebook_df.columns:
        codebook_df[val_label_col] = ''
    
    # 변수 레이블 결정
    if variable_label:
        new_var_label = variable_label
    else:
        # 원본 변수의 레이블 가져오기
        source_rows = codebook_df[codebook_df[var_col] == source_variable]
        if not source_rows.empty:
            first_label = source_rows.iloc[0][var_label_col]
            if pd.notna(first_label) and str(first_label).strip():
                new_var_label = f"{first_label} (변환됨)"
            else:
                new_var_label = f"{target_variable} (변환됨)"
        else:
            new_var_label = f"{target_variable} (변환됨)"
    
    # 기존에 이 변수가 있으면 삭제
    codebook_df = codebook_df[codebook_df[var_col] != target_variable]
    print(f"기존 {target_variable} 제거 후 행 수: {len(codebook_df)}")
    
    # 새 행 생성
    new_rows = []
    processed_values = set()
    
    for rule in rules:
        new_value = str(rule.get('new', '')).strip()
        value_label = str(rule.get('label', '')).strip()
        
        if new_value and new_value not in processed_values:
            processed_values.add(new_value)
            
            # 새 행 생성 - 기존 컬럼 순서 유지
            new_row = {}
            for col in codebook_df.columns:
                if col == var_col:
                    new_row[col] = target_variable
                elif col == var_label_col:
                    # 첫 번째 행에만 변수 레이블
                    new_row[col] = new_var_label if len(new_rows) == 0 else ''
                elif col == val_col:
                    new_row[col] = new_value
                elif col == val_label_col:
                    # 값 레이블 (사용자 입력이 있으면 사용, 없으면 값 자체)
                    new_row[col] = value_label if value_label else new_value
                else:
                    new_row[col] = ''
            
            new_rows.append(new_row)
            print(f"추가할 행: {new_row}")
    
    # 새 행 추가
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        
        # 컬럼 순서 명시적으로 지정: Variable, Variable Label, Value, Value Label, 나머지
        ordered_columns = []
        
        # 1. Variable
        if var_col in new_df.columns:
            ordered_columns.append(var_col)
        
        # 2. Variable Label
        if var_label_col in new_df.columns:
            ordered_columns.append(var_label_col)
        
        # 3. Value
        if val_col in new_df.columns:
            ordered_columns.append(val_col)
        
        # 4. Value Label
        if val_label_col in new_df.columns:
            ordered_columns.append(val_label_col)
        
        # 5. 나머지 컬럼들
        for col in new_df.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        # 순서 적용
        new_df = new_df[ordered_columns]
        
        # 기존 코드북도 같은 순서로 재정렬
        codebook_df = codebook_df[ordered_columns]
        
        # 결합
        codebook_df = pd.concat([codebook_df, new_df], ignore_index=True)
        print(f"업데이트 후 행 수: {len(codebook_df)}")
        print(f"최종 컬럼 순서: {codebook_df.columns.tolist()}")
        print(f"추가된 행:\n{new_df}")
    
    # 코드북 파일 저장
    if codebook.file.name.endswith('.csv'):
        codebook_df.to_csv(codebook.file.path, index=False, encoding='utf-8-sig')
        print(f"CSV로 저장: {codebook.file.path}")
    else:
        codebook_df.to_excel(codebook.file.path, index=False, engine='openpyxl')
        print(f"Excel로 저장: {codebook.file.path}")
    
    print(f"=== 코드북 업데이트 완료: {target_variable} ({len(new_rows)}개 행 추가) ===")

def remove_variable_from_codebook(codebook, variable):
    """코드북에서 특정 변수 제거"""
    # 코드북 로드
    if codebook.file.name.endswith('.csv'):
        codebook_df = pd.read_csv(codebook.file.path, encoding='utf-8-sig')
    else:
        codebook_df = pd.read_excel(codebook.file.path)
    
    # 필수 컬럼 확인
    if 'variable' not in codebook_df.columns:
        return
    
    # 해당 변수 제거
    original_count = len(codebook_df)
    codebook_df = codebook_df[codebook_df['variable'] != variable]
    removed_count = original_count - len(codebook_df)
    
    # 코드북 파일 저장
    if codebook.file.name.endswith('.csv'):
        codebook_df.to_csv(codebook.file.path, index=False, encoding='utf-8-sig')
    else:
        codebook_df.to_excel(codebook.file.path, index=False, engine='openpyxl')
    
    print(f"코드북에서 {variable} 변수 제거 완료 ({removed_count}개 행 삭제)")

def create_codebook_from_spss(survey_data):
    """SPSS 파일에서 변수 설명과 값 라벨을 추출하여 코드북 생성"""
    if not pyreadstat:
        raise ImportError("pyreadstat module is not installed.")
        
    df, meta = pyreadstat.read_sav(survey_data.file.path)
    
    rows = []
    
    # 변수 순회
    for col_name in meta.column_names:
        var_label = meta.column_names_to_labels.get(col_name, "")
        val_labels = meta.variable_value_labels.get(col_name, {})
        
        if val_labels:
            # 값 라벨이 있는 경우 각 값마다 행 생성
            first_row = True
            for val, label in val_labels.items():
                rows.append({
                    'Variable': col_name,
                    'Variable Label': var_label if first_row else "",
                    'Value': val,
                    'Value Label': label
                })
                first_row = False
        else:
            # 값 라벨이 없으면 변수 정보만 한 줄 추가
            rows.append({
                'Variable': col_name,
                'Variable Label': var_label,
                'Value': "",
                'Value Label': ""
            })
            
    # DataFrame 생성
    codebook_df = pd.DataFrame(rows)
    
    # 코드북 파일 저장 (Excel)
    codebook_filename = os.path.splitext(os.path.basename(survey_data.file.name))[0] + '_codebook.xlsx'
    codebook_path = os.path.join(settings.MEDIA_ROOT, 'codebooks', codebook_filename)
    
    # 디렉토리 확인
    os.makedirs(os.path.dirname(codebook_path), exist_ok=True)
    
    codebook_df.to_excel(codebook_path, index=False, engine='openpyxl')
    
    # Codebook 모델 생성
    Codebook.objects.create(
        user=survey_data.user,
        dataset=survey_data,
        name=f"{survey_data.name} (Auto Codebook)",
        file=os.path.join('codebooks', codebook_filename),
        description="SPSS 파일에서 자동으로 추출된 코드북입니다."
    )
