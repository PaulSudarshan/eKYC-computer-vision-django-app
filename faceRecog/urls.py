"""faceRecog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from recog import views as recog_views
from records import views as record_views
from django.urls import path
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', recog_views.home, name='home'),
    path('dashboard/', recog_views.dashboard, name='dashboard'),
    
    
    url(r'^error_image$', recog_views.errorImg),
    url(r'^dashboard/create_dataset$', recog_views.create_dataset),
    url(r'^dashboard/trainer$', recog_views.trainer),
    url(r'^dashboard/eigen_train$', recog_views.eigenTrain),
    url(r'^dashboard/detect$', recog_views.detect),
    url(r'^dashboard/detect_image$', recog_views.detectImage),
    url(r'^admin/', admin.site.urls),
    
    
    url(r'^records/details/(?P<id>[\w.@+-]+)/$', record_views.details, name='details'),
    #url(r'^records', record_views.index, name='index'),
    path('register/', record_views.register, name='register'),
    
    path('login/',auth_views.LoginView.as_view(template_name='records/login.html'),name='login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='recog/home.html'),name='logout'),
    path('not_authorised', recog_views.not_authorised, name='not-authorised'),
    
    path('signup/', record_views.signup_view, name="signup")
]
