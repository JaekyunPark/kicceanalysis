from django import forms
from .models import SurveyData, Codebook

class SurveyDataUploadForm(forms.ModelForm):
    class Meta:
        model = SurveyData
        fields = ['name', 'file', 'description']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'file': forms.FileInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

class CodebookUploadForm(forms.ModelForm):
    class Meta:
        model = Codebook
        fields = ['name', 'dataset', 'file', 'description']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'dataset': forms.Select(attrs={'class': 'form-select'}),
            'file': forms.FileInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }
    
    def __init__(self, user=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            # 현재 사용자의 데이터셋만 표시
            self.fields['dataset'].queryset = SurveyData.objects.filter(user=user)
        
        # dataset 필드를 선택사항으로 설정
        self.fields['dataset'].required = False
        self.fields['dataset'].empty_label = "데이터셋 선택 안함 (범용 코드북)"
