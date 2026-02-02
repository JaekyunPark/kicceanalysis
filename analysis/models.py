from django.db import models
from django.conf import settings
from surveys.models import SurveyData

class AnalysisPreset(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    dataset = models.ForeignKey(SurveyData, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    configuration = models.JSONField(help_text="분석 설정 (변수, 통계량, 옵션 등)")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.dataset.name}"

    class Meta:
        ordering = ['-created_at']
