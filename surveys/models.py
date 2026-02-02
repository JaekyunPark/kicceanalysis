from django.db import models
from django.contrib.auth.models import User

class SurveyData(models.Model):
    """설문조사 데이터 파일"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='survey_data/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Survey Data'
        verbose_name_plural = 'Survey Data'
    
    def __str__(self):
        return self.name

class Codebook(models.Model):
    """코드북 (변수 및 값 레이블)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset = models.ForeignKey(
        SurveyData, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='codebooks',
        help_text='이 코드북과 연결된 데이터셋 (선택사항)'
    )
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='codebooks/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Codebook'
        verbose_name_plural = 'Codebooks'
    
    def __str__(self):
        return self.name

class AnalysisResult(models.Model):
    """분석 결과 저장"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset = models.ForeignKey(SurveyData, on_delete=models.CASCADE)
    codebook = models.ForeignKey(Codebook, on_delete=models.SET_NULL, null=True, blank=True)
    analysis_type = models.CharField(max_length=50)  # 'crosstab', 'frequency', etc.
    created_at = models.DateTimeField(auto_now_add=True)
    parameters = models.JSONField()  # 분석에 사용된 파라미터
    result_file = models.FileField(upload_to='analysis_results/', null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis Result'
        verbose_name_plural = 'Analysis Results'
    
    def __str__(self):
        return f"{self.analysis_type} - {self.dataset.name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class RecodeRule(models.Model):
    """변수 변환(RECODE) 규칙"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset = models.ForeignKey(SurveyData, on_delete=models.CASCADE, related_name='recode_rules')
    name = models.CharField(max_length=200, help_text='규칙 이름 (예: 성별 숫자→텍스트)')
    source_variable = models.CharField(max_length=100, help_text='원본 변수명')
    target_variable = models.CharField(max_length=100, help_text='새 변수명')
    rules = models.JSONField(help_text='변환 규칙들 (JSON)')
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Recode Rule'
        verbose_name_plural = 'Recode Rules'
        unique_together = [['user', 'dataset', 'target_variable']]
    
    def __str__(self):
        return f"{self.name} ({self.source_variable} → {self.target_variable})"
