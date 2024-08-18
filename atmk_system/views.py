from django.contrib import auth  # 引入auth模块
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from atmk_system.utils import response_success, response_error
import json


@require_POST
@csrf_exempt
def login(request):
    data = json.loads(request.body)
    username = data.get('username')
    password = data.get('password')
    # 调用auth.authenticate()方法进行登录校验
    user_obj = auth.authenticate(username=username, password=password)
    if user_obj:
        # 校验成功，调用auth.login（request, user_obj）方法：
        auth.login(request, user_obj)
        return response_success(data={})
    else:
        return response_error('login error')


@require_POST
def logout(request):
    # 调用 auth.logout(request)方法，类似session的request.session.flush()同时删除session表记录和cookie
    auth.logout(request)
    return JsonResponse({'status': 1})


@login_required
def user_info(request):
    return response_success(data={
        'username': request.user.username
    })
