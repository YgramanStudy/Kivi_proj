from rest_framework import serializers
from .models import *


# class ArticleSerializer(serializers.Serializer):
#     title = serializers.CharField(max_length=64)
#     body = serializers.CharField()
#     date = serializers.DateTimeField('date published')
#
#
#     def create(self, validated_data):
#         return Article.objects.create(validated_data)
#
#     def update(self, instance, validated_data):
#         instance.title = validated_data.get('title', instance.title)
#         instance.body = validated_data.get('body', instance.body)
#         instance.date = validated_data.get('date', instance.date)

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'body', 'date']

class DogPicSerializer(serializers.ModelSerializer):
    class Meta:
        model = DogPic
        fields = ['id', 'dog', 'pic', 'deleted']

