from django.urls import path
from . import views

app_name = 'surveys'

urlpatterns = [
    path('', views.home, name='home'),
    path('data/upload/', views.data_upload, name='data_upload'),
    path('data/list/', views.data_list, name='data_list'),
    path('data/delete/<int:pk>/', views.data_delete, name='data_delete'),
    path('codebook/upload/', views.codebook_upload, name='codebook_upload'),
    path('codebook/list/', views.codebook_list, name='codebook_list'),
    path('codebook/delete/<int:pk>/', views.codebook_delete, name='codebook_delete'),
    
    # 변수 변환 (RECODE)
    path('recode/', views.recode_variable, name='recode'),
    path('recode/preview/', views.recode_preview, name='recode_preview'),
    path('recode/apply/', views.recode_apply, name='recode_apply'),
    path('recode/delete/<int:rule_id>/', views.recode_delete, name='recode_delete'),
    
    # API 엔드포인트
    path('api/variables/<int:dataset_id>/', views.get_variables_for_dataset, name='get_variables'),
    path('api/variable-values/<int:dataset_id>/<str:variable>/', views.get_variable_values, name='get_variable_values'),
]
