# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .models import Records
from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.shortcuts import render, redirect

from .forms import SignupForm
from .models import Records
import os

# Create your views here.
def index(request):
    records = Records.objects.all()[:10]    #getting the first 10 records
    
    context = {
        'records': records
    }
    return render(request, 'records.html', context)

def details(request, id):
    record = Records.objects.get(id=id)
    print('DETAILS')
    context = {
        'record' : record
    }
    
    return render(request, 'details.html', context)
    

@login_required
def register(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=UserCreationForm(request.POST)
		if form.is_valid():
			form.save() ###add user to database
			messages.success(request, f'Employee registered successfully!')
			return redirect('')
		


	else:
		form=UserCreationForm()
	return render(request,'records/register.html', {'form' : form})
	
def handle_uploaded_file(f,name_id):
    print(os.getcwd())
    print(os.listdir())
    destination = open('static/users/'+name_id+f.name.split('.')[-1], 'wb+')
    for chunk in f.chunks():
        destination.write(chunk)
        destination.close()
        
@login_required
def signup_view(request):
    if request.method=='POST':
        signform = SignupForm(request.POST,request.FILES)
        if signform.is_valid():
            
            #handle_uploaded_file(request.FILES['avatar'],signform.cleaned_data.get('id'))
            signform.save()
    else:
        signform=SignupForm()
        
    context = {
            'signform_key':signform
            }
    return render(request,'records/signup.html',context)
