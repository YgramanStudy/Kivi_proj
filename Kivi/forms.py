from django.forms import ModelForm, TextInput
from .models import *
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
# from django.db.models.functions import Now
from datetime import date, datetime, timedelta


from models import Profile


class ProfileForm(forms.ModelForm):
    first_name = forms.CharField(max_length=255)
    last_name = forms.CharField(max_length=255)
    email = forms.EmailField()

    class Meta:
        model = Profile
        fields = '__all__'
        exclude = ['user']


def form_validation_error(form):
    msg = ""
    for field in form:
        for error in field.errors:
            msg += "%s: %s \\n" % (field.label if hasattr(field, 'label') else 'Error', error)
    return msg
















class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = ['title', 'body']

    def save(self, commit=True):
        article = super(ArticleForm, self).save(commit=False)

        if commit:
            article.date = datetime.now()
            article.save()
        return


class StudentUserForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("first_name", "last_name", "username", "email", "password1", "password2")

    def save(self, commit=True):
        user = super(StudentUserForm, self).save(commit=False)  # create user in auth table
        user.email = self.cleaned_data["email"]
        if commit:
            user.is_staff = False
            user.save()
        return user