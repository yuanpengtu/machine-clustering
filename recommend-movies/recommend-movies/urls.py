from django.conf.urls import include, url

from django.contrib import admin
admin.autodiscover()

import app.views

# Examples:
# url(r'^$', 'gettingstarted.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),

urlpatterns = [
    url(r'^$', app.views.index, name='index'),
    #url(r'^cluster/$', app.views.cluster, name='cluster'),
    url(r'^id/$', app.views.getid, name='getid'),
    url(r'^movies/$', app.views.movies, name='movies'),
    url(r'^recommend/$', app.views.recommend, name='recommend'),
    url(r'^db', app.views.db, name='db'),
    url(r'^admin/', include(admin.site.urls)),
]
