from django import forms
from .models import Records


class SignupForm(forms.ModelForm):
    class Meta:
        model = Records        
        fields=['id','first_name','last_name','residence','country','education','occupation','marital_status','bio','recorded_at','avatar']
