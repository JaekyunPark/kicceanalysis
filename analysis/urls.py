from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [
    # 통합 분석
    path('', views.unified_analysis, name='unified_analysis'),
    
    # 엑셀 내보내기
    path('export-excel/', views.export_analysis_to_excel, name='export_excel'),
    
    # API 엔드포인트
    path('get-codebooks/<int:dataset_id>/', views.get_codebooks_for_dataset, name='get_codebooks'),
    path('get-variables/<int:dataset_id>/', views.get_variables_for_dataset, name='get_variables'),
]
