from django.contrib import admin
from .models import SurveyData, Codebook, AnalysisResult

@admin.register(SurveyData)
class SurveyDataAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'uploaded_at']
    list_filter = ['uploaded_at', 'user']
    search_fields = ['name', 'description']

@admin.register(Codebook)
class CodebookAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset', 'user', 'uploaded_at']
    list_filter = ['uploaded_at', 'user']
    search_fields = ['name', 'description']

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ['analysis_type', 'dataset', 'user', 'created_at']
    list_filter = ['analysis_type', 'created_at', 'user']
    search_fields = ['dataset__name']
    readonly_fields = ['created_at']
