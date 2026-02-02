from django import template
from django.utils.html import format_html
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def index(sequence, position):
    """리스트나 튜플의 특정 위치 요소 반환"""
    try:
        return sequence[position]
    except (IndexError, TypeError, KeyError):
        return ''

@register.filter
def truncate_with_tooltip(text, max_length=30):
    """긴 텍스트를 축약하고 툴팁으로 전체 텍스트 표시"""
    text = str(text)
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length] + '...'
    return format_html(
        '<span class="long-text" data-full-text="{}">{}</span>',
        text,
        truncated
    )

@register.filter
def format_stat_value(value, stat_type=''):
    """통계값 포맷팅 (평균은 소수점 2자리, 나머지는 그대로)"""
    try:
        num_value = float(value)
        # 평균 관련 통계는 소수점 2자리
        if '평균' in stat_type or 'mean' in stat_type.lower():
            return f'{num_value:.2f}'
        # 표준편차, 중앙값 등도 소수점 2자리
        elif any(keyword in stat_type.lower() for keyword in ['std', '표준편차', 'median', '중앙값', 'q1', 'q3']):
            return f'{num_value:.2f}'
        # 최솟값, 최댓값은 원래 형태 유지
        else:
            return f'{num_value:.2f}'
    except (ValueError, TypeError):
        return value
