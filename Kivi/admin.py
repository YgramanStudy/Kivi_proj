from django.contrib import admin
from .models import *

# Register your models here.

# admin.site.register(Profile)

@admin.register(Profile)
class ProfileModel(admin.ModelAdmin):
    list_filter = ('gender', 'created_at')

# admin.site.register(Article)
@admin.register(Article)
class ArticleModel(admin.ModelAdmin):
    list_filter = ('title', 'body')
    list_display = ('title', 'body')

admin.site.register(Dog)
admin.site.register(UserDog)
admin.site.register(DogPic)
admin.site.register(DogsRelation)

# admin.site.register()
# admin.site.register()
# admin.site.register()
# admin.site.register()
# admin.site.register()
# admin.site.register()
# admin.site.register()
