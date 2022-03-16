from django.urls import path, include
from .views import *
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register('articles', ArticleViewSet, basename='articles')
router.register('dogpic', DogPicViewSet, basename='dogpic')

urlpatterns = [
    path('api/', include(router.urls)),
    # path('', include(router.urls)),

    # path('articles/', ArticleList.as_view()),
    # path('articles/<int:id>/', ArticleDetails.as_view()),
    # 01:05
    # path('articles/', article_list),
    # path('articles/<int:pk>/', article_details),
]
