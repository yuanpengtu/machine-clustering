from django.db import models

# Create your models here.
class Greeting(models.Model):
    when = models.DateTimeField('date created', auto_now_add=True)


class User(models.Model):
	userid = models.IntegerField()
	age = models.IntegerField()
	sex = models.CharField(max_length=1)
	occupation = models.CharField(max_length=50)
	zipcode = models.IntegerField(max_length=20) 
	avg = models.FloatField(default=0.0)

class Movie(models.Model):
	movieid = models.IntegerField()
	title = models.CharField(max_length=100)
	date = models.CharField(max_length=100)
	viddate = models.CharField(max_length=100)
	url = models.CharField(max_length=500)
	unknown = models.IntegerField(max_length=1)
	action = models.IntegerField(max_length=1)
	adventure = models.IntegerField(max_length=1)
	animation = models.IntegerField(max_length=1)
	childrens = models.IntegerField(max_length=1)
	comedy = models.IntegerField(max_length=1)
	crime = models.IntegerField(max_length=1)
	documentary = models.IntegerField(max_length=1)
	drama = models.IntegerField(max_length=1)
	fantasy = models.IntegerField(max_length=1)
	film_noir = models.IntegerField(max_length=1)
	horror = models.IntegerField(max_length=1)
	musical = models.IntegerField(max_length=1)
	mystery = models.IntegerField(max_length=1)
	romance = models.IntegerField(max_length=1)
	sci_fi = models.IntegerField(max_length=1) 
	thriller = models.IntegerField(max_length=1)
	war = models.IntegerField(max_length=1)
	western = models.IntegerField(max_length=1)

class Rating(models.Model):
	userid = models.IntegerField()
	movieid = models.IntegerField()
	rating = models.IntegerField()
	time = models.IntegerField(max_length=9)

class Cluster(models.Model):
	name = models.CharField(max_length=100)
	users = models.ManyToManyField(User)

	def get_members(self):
		return "\n".join([u.username for u in self.users.all()])
