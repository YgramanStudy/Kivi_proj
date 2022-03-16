from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator
from django.contrib.auth.forms import UserChangeForm
from datetime import date, datetime


from django.utils.translation import gettext as _
# from django.contrib.staticfiles.templatetags.staticfiles import static




class Article(models.Model):
    title = models.CharField(max_length=64)
    body = models.TextField()
    date = models.DateTimeField('date published')

    def str(self):
        return self.title

    class Meta:
        # for admin
        verbose_name = "Article"
        verbose_name_plural = "Articles"



class Profile(models.Model):
    GENDER_MALE = 1
    GENDER_FEMALE = 2
    GENDER_CHOICES = [
        (GENDER_MALE, _("Male")),
        (GENDER_FEMALE, _("Female")),
    ]

    user = models.OneToOneField(User, related_name="profile", on_delete=models.CASCADE)
    # avatar = models.ImageField(upload_to="customers/profiles/avatars/", null=True, blank=True)
    birthday = models.DateField(null=True, blank=True)
    gender = models.PositiveSmallIntegerField(choices=GENDER_CHOICES, null=True, blank=True)

    phone = models.CharField(max_length=32, null=True, blank=True)
    phone_2 = models.CharField(max_length=32, null=True, blank=True)
    home_phone = models.CharField(max_length=32, null=True, blank=True)

    address = models.CharField(max_length=255, null=True, blank=True)
    number = models.CharField(max_length=32, null=True, blank=True)
    city = models.CharField(max_length=50, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _('Profile')
        verbose_name_plural = _('Profiles')

    def __str__(self):
        return self.user.username + " Profile"

    # @property
    # def get_avatar(self):
    #     return self.avatar.url if self.avatar else static('assets/img/team/default-profile-picture.png')


class Dog(models.Model):
    GENDER_MALE = 1
    GENDER_FEMALE = 2
    GENDER_CHOICES = [
        (GENDER_MALE, _("Male")),
        (GENDER_FEMALE, _("Female")),
    ]

    GOOD = 1
    REGULAR = 2
    BAD = 3
    BEHAVIOR_CHOICES = [
        (GOOD, _("GOOD")),
        (REGULAR, _("REGULAR")),
        (BAD, _("BAD")),
    ]

    name = models.CharField(max_length=64, null=True, blank=True)
    # avatar = models.ImageField(upload_to="customers/profiles/avatars/", null=True, blank=True)
    gender = models.PositiveSmallIntegerField(choices=GENDER_CHOICES, null=True, blank=True)
    birthday = models.DateField(null=True, blank=True)
    behavior = models.PositiveSmallIntegerField(choices=BEHAVIOR_CHOICES, null=True, blank=True, default=REGULAR)

    class Meta:
        verbose_name = _('Dog')
        verbose_name_plural = _('Dogs')

    def __str__(self):
        return "ID: " + str(self.id) + " Name: " + self.name



class DogPic(models.Model):
    dog = models.ForeignKey(Dog, null=True, on_delete=models.CASCADE)
    pic = models.ImageField(upload_to="dog_photos/", null=True, blank=True)
    # should be :/media/dog_photos/dog_id/

    deleted = models.BooleanField(default=False)

    # pic = models.ManyToManyField()

class UserDog(models.Model):
    owner = models.ForeignKey(User, null=True, on_delete=models.CASCADE)
    dog = models.ForeignKey(Dog, null=True, on_delete=models.CASCADE)

    def __str__(self):
        return "Owner: " + self.owner.username + "     Dog: " + self.dog.name


class DogsRelation(models.Model):
    GOOD = 9
    NEUTRAL = 5
    BAD = 1
    RELATION_CHOICES = [
        (GOOD, _("Good")),
        (NEUTRAL, _("NEUTRAL")),
        (BAD, _("Bad")),
    ]

    ow_dog = models.ForeignKey(Dog,  related_name='owner_dog', on_delete=models.CASCADE)
    ot_dog = models.ForeignKey(Dog, related_name='other_dog', on_delete=models.CASCADE)

    relation = models.PositiveSmallIntegerField(choices=RELATION_CHOICES, null=True, blank=True)


# class LostListPic(models.Model):
#     datetime = models.DateTimeField()
#     pic = models.ImageField()
#     # location = models.
#     # found_by = models.CharField()
#
#
# class LostDog(models.Model):
#     pass
# #   one to many dog vs lostpic
# #   auto delete if found also from LostListPic
# #   choosing avatar to represent the dog
# #   auto adding new pic to lost dog
# #   auto sending notification to dog owner if found


