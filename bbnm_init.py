from flask import Flask, request, render_template, flash, url_for, redirect, session, jsonify, g
from flask_login import LoginManager , login_required
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import *
from sqlalchemy import DateTime, Column, Integer, String, create_engine, Sequence, text, MetaData, Table, inspect, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, mapper, relationship, scoped_session, create_session
from sqlalchemy.sql.expression import bindparam

from datetime import datetime

from passlib.hash import sha256_crypt

from getpass import getpass
import sys

from flask import current_app
import gc
from functools import wraps
import pandas as pd
from sqlalchemy import and_
from json import dumps, loads, JSONEncoder, JSONDecoder
from wtforms import Form, BooleanField, TextField, PasswordField, validators
import gc
import numpy as np
import os
import random, copy
from random import shuffle
import csv
from datetime import datetime
import pytz
import itertools
import ast

#for def rec_3names
import scipy.linalg
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from subprocess import check_output
import espeak
from functools import reduce
import scipy as sp





bbnm = Flask(__name__)
bbnm.secret_key = "jiad00ad8-0aigj:edomv,d,afjldjkd1999iuoijwo;ej87342k"


engine = create_engine('sqlite:///bbnm.db')
db_session = scoped_session(sessionmaker(autocommit=False,autoflush=False,bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()
Session = sessionmaker(bind=engine)
db = Session()
metadata = MetaData()
inspector = inspect(engine)
meta = MetaData(bind=engine)

Base.metadata.create_all(engine)
sqlbn = SQLAlchemy(bbnm)

login_manager = LoginManager()
login_manager.init_app(bbnm)
login_manager.login_view='login'


class User(Base):
    __tablename__='users'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, Sequence('user_id_seq'),primary_key=True)
    username = Column(String(20), unique=True, index=True)
    password = Column(String(30))
    email = Column(String(50), unique=True, index=True)
    registered_on = Column(DateTime)
    

    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email
        self.registered_on = datetime.utcnow()
    
    def is_authenticated(self):
        return True    

    def is_active(self):
        """ True, as all users are active. """
        return True
    
    def is_anonymous(self):
        """False, as is_anonymous users aren't supported."""
        return False
    
    def get_id(self):
        return (self.id)
    
    def __repr__(self):
        return "<User(username='%s', password='%s', email='%s', registered_on = '%s')>" %(self.username, self.password, self.email, self.registered_on)
    
class Searchlist(Base):
    __tablename__= 'search_lists'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    search_name = Column(String, nullable=False)
    usr_search_name = Column(Text)
    date = Column(DateTime)
    description = Column(Text)
    
    def __init__(self, user_id, search_name, usr_search_name, description):
        self.user_id = user_id
        self.search_name = search_name
        self.usr_search_name = usr_search_name
        self.date = datetime.utcnow()
        self.description = description
    def __repr__(self):
        return "<%s(%s,%s,%s,%s,%s,%s)>" % (Searchlist, self.id, self.user_id, self.search_name, self.usr_search_name, self.date, self.description)

class UserRatings(Base):
    __tablename__ = 'userratings'
    __table_args__={'extend_existing': True}
    id = Column(Integer, primary_key=True)
    searchlist_id = Column(Integer,ForeignKey('search_lists.id'))
    user_id = Column(Integer,ForeignKey('users.id'),default="NAN")
    name = Column(String)
    sex = Column(String)
    rating = Column(String,default="NAN")
    saved = Column(String, default="NAN")

    def __repr__(self):
        return "<UserRatings(%s,%s,%s,%s,%s,%s)>" % (self.id, self.searchlist_id, self.user_id, self.name, self.rating, self.saved)
    
Base.metadata.create_all(engine)
db._model_changes={}
db.commit()


#pulls first 3 similar names
def rec_5names(name):
    filename = '/home/airos/flask-web/babyname/names_ipa.csv'
    nl_df = pd.read_csv(filename)
    names_ipa = []
    nl_df['ipa'].apply(lambda x: names_ipa.append(x))
    #vectorizer = CountVectorizer(analyzer='char',decode_error='ignore',min_df=1, max_df=.9)
    #transformer = TfidfTransformer(smooth_idf=True)
    vectorizer = TfidfVectorizer(analyzer='char',decode_error='ignore',min_df=1, max_df=.9)
    
    #namesounds_vectors = vectorizer.fit_transform(names_ipa)
    #tfidf_allnames = transformer.fit_transform(namesounds_vectors)
    #tfidf_allnames.toarray()
    
    tfidf_allnames = vectorizer.fit_transform(names_ipa)
    num_clusters = 30
    km = KMeans(n_clusters=num_clusters, init='k-means++', n_init=1, verbose=1, random_state=3)
    km.fit(tfidf_allnames)
    newname = name
    #may make processing slow
    newname_ipa = check_output(["espeak","-q","--ipa",'-v','en-us', newname]).decode('utf-8')
    newname_vectorize = vectorizer.transform([newname_ipa])
    newname_label = km.predict(newname_vectorize)[0]
    similar_indices = (km.labels_==newname_label).nonzero()[0]
    similar = []
    for i in similar_indices:
        dist = sp.linalg.norm((newname_vectorize-tfidf_allnames[i]).toarray())
        similar.append((dist,nl_df['name'][i]))
    similar = sorted(similar)
    top5 = []
    for item in range(len(similar)):
        if item == 0:
            top5.append(similar[item])
        elif check_output(["espeak","-q","--ipa",'-v','en-us', similar[item][1]]).decode('utf-8') != check_output(["espeak","-q","--ipa",'-v','en-us', top5[-1][1]]).decode('utf-8'):
            top5.append(similar[item])
            if len(top5) == 20:
                break
    return top5

#loads each user's searches and saves their searchlist_id here, search name (user vs programming side) and search description
def usr_searches(user_id, search_name, usr_search_name, description):
    Base.metadata.create_all(engine)
    meta.create_all()
    db.execute(Searchlist.__table__.insert().values(user_id = user_id, search_name = search_name, usr_search_name = usr_search_name, description = description, date = datetime.utcnow()))
    db._model_changes={}
    db.commit()
    return (Searchlist)

#loads names and searchlist_id into the UserRatings table: unrated names have ratings ='NAN' (will update responses and pull unrated names from this table)
def start_search(survey_id,user_id):
    namelist = pd.read_csv('names_start.csv')
    namelist = namelist[['name','sex']]
    namedict = namelist.set_index('name')['sex'].to_dict()
    db.add_all([
        UserRatings(
            searchlist_id=survey_id,
            user_id=user_id,
            name='%s' %i,
            sex='%s' %v)
        for i,v in namedict.items() 
        ])
    db._model_changes={}
    db.commit()
    return(UserRatings)

#updates the UserRatings table with dictionaries of user responses for names (ratings and names they want to save)
def add_usrsearch_input(survey_id, user_ratings, user_saved):
    for key,value in user_ratings.items():
        db.execute(UserRatings.__table__.update().where(and_(UserRatings.searchlist_id == survey_id, UserRatings.name == '%s' %key)).values(rating=bindparam('rating')), 
                   [
            {'rating':value}
            ])
    for key,value in user_saved.items():
        db.execute(UserRatings.__table__.update().where(and_(UserRatings.searchlist_id == survey_id, UserRatings.name == '%s' %key)).values(saved='1'))
    db._model_changes={}
    db.commit()
    return(UserRatings)



def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash("You need to login first")
            return redirect(url_for('login'))
    return wrap    

@login_manager.user_loader
def user_loader(user_id):
    """Given *user_id*, return the associated User object. 
    :param unicode user_id: user_id (id) user to retrieve
    """
    #Ids in Flask-Login = unicode strings
    #have to convert to integer before querying using SQLAlchemy
    return db.query(str(id))

@bbnm.route('/')
def index():
    flash('Please login or register')
    return redirect(url_for('login'))

@bbnm.route('/register', methods = ['GET','POST'])
def register():
    if request.method == 'GET':
        return render_template('bbnmregister.html')
    if request.form['password'] != request.form['verify_password']:
        flash("Passwords do not match. Try again")
        session['logged_in']=False
        return(redirect(url_for('register')))
    username = request.form['username']
    data = db.query(User).from_statement(text("SELECT * FROM users where username=:username")).params(username=username).all()
    if len(data) != 0:
        flash("Username has already been taken. Please choose another.")
        return(redirect(url_for('register')))
    password = sha256_crypt.hash(str(request.form['password']))
    email = request.form['email']
    user = User(username, password, email)
    db.add(user)
    db._model_changes = {}
    db.commit()
    session['logged_in']=True
    session['username']=username
    flash('Thank you for registering! We hope you enjoy the Babyname Recommender.')
    return render_template('bbnmhome.html', username=username)

@bbnm.route('/login', methods = ['GET','POST'])
def login():
    """For GET requests, display the login form. For POSTS, login the current user by processing the form.
    """
    if request.method == 'GET':
        return render_template('bbnmlogin.html')
    username = request.form['username']
    data = db.query(User).from_statement(text("SELECT * FROM users where username=:username")).params(username=username).all()
    if len(data) == 1:
        password = data[0].password
        password2 = str(request.form['password'])
        if sha256_crypt.verify(password2,password):
            registered_user =db.query(User).filter_by(username=username).first()
            session['logged_in']=True
            session['username']=username
            flash("You are now logged in")
            return redirect(request.args.get('next') or url_for('home', username = username))
        else:
            error = "Invalid credentials. Try again."
            session['logged_in']=False
            flash('Username or Password is invalid', error)
            return(render_template('bbnmlogin.html'))
    else:
        flash('Username or Password is invalid', 'error')
        return(render_template('bbnmlogin.html'))

@bbnm.route('/new_search',methods=['GET','POST'])
@login_required
def new_search():
    username = session['username']
    user = db.query(User).filter_by(username=username).one()
    user_id = str(user.id)
    usr_search_name = request.form['usr_search_name']
    search_description = request.form['search_description']
    usersearches = db.query(Searchlist.usr_search_name).filter_by(user_id=user_id).all()
    usersearches = list(itertools.chain(*usersearches))
    if len(usersearches) > 0:
        if usr_search_name in usersearches:
            flash("You have already started a search with this name. Please choose another")
            return(redirect(url_for('searches')))
        else:
            searchnumber = len(usersearches)
            searchnumber = str(searchnumber+1)
            search_name = 'user'+user_id+'_search'+searchnumber
            savesearch = usr_searches(user_id, search_name, usr_search_name, search_description)
            searchlist_id = list(db.query(Searchlist).filter_by(search_name=search_name))[0].id
            new_search = start_search(searchlist_id,user_id)
            namelist = db.query(UserRatings.name).filter_by(rating="NAN",searchlist_id=searchlist_id).all()
            namelist = list(itertools.chain(*namelist))
            shuffle(namelist)
            return render_template('bbnmratenames.html',namelist = namelist, search_name = search_name)
    else:
        search_name = 'user'+user_id+'_search1'
        savesearch = usr_searches(user_id, search_name, usr_search_name,search_description)
        searchlist_id = list(db.query(Searchlist).filter_by(search_name=search_name))[0].id
        new_search = start_search(searchlist_id,user_id)
        namelist = db.query(UserRatings.name).filter_by(rating="NAN",searchlist_id=searchlist_id).all()
        namelist = list(itertools.chain(*namelist))
        shuffle(namelist)
        return render_template('bbnmratenames.html',namelist = namelist, search_name = search_name)


@bbnm.route('/searches', methods = ['GET','POST'])
@login_required
def searches():
    if request.method == 'GET':
        username = session['username']
        user = db.query(User).from_statement(text("SELECT * FROM users where username=:username")).params(username=username).one()
        user_id = str(user.id)
        
        if len(list(db.query(Searchlist.search_name))) > 0:
            searchlist = db.query(Searchlist.search_name, Searchlist.usr_search_name, Searchlist.description, Searchlist.date).filter_by(user_id = user_id).all()
            if len(searchlist) > 0:
                return render_template('bbnmsearch.html', searchlist=searchlist)
            else:
                return render_template('bbnmsearch.html', searchlist = None)
        else:
            return render_template('bbnmsearch.html', searchlist = None)
    else:
        search_name = request.form['search_name']
        return redirect(url_for('ratenames',search_name=search_name))


@bbnm.route('/_ratednames', methods = ['POST'])
@login_required
def _ratednames():
    
    namedata = request.json['names_ratings']
    search_name = request.json['search_name']
    saved_names = request.json['saved_names']
    
    searchlist_id = list(db.query(Searchlist).filter_by(search_name=search_name))[0].id
    adduserdata = add_usrsearch_input(searchlist_id,namedata,saved_names)
    result = "Names and ratings saved!"
    return jsonify(result)

@bbnm.route('/ratenames/<search_name>', methods = ['GET','POST'])
@login_required
def ratenames(search_name):
    searchlist_id = list(db.query(Searchlist).filter_by(search_name=search_name))[0].id
    namelist = db.query(UserRatings.name).filter_by(rating="NAN",searchlist_id=searchlist_id).all()
    namelist = list(itertools.chain(*namelist))
    shuffle(namelist)
    return render_template('bbnmratenames.html', namelist=namelist, search_name = search_name)
    
@bbnm.route('/logout', methods=['GET'])
@login_required
def logout():
    """Logout the current user.
    """
    session.clear()
    flash('You have been logged out')
    return redirect(url_for('login'))

@bbnm.route('/home/<username>')
@login_required
def home(username):
    return render_template('bbnmhome.html', username=username)


@bbnm.route('/savednames')
@login_required
def savednames():
    username = session['username']
    user = db.query(User).from_statement(text("SELECT * FROM users where username=:username")).params(username=username).one()
    user_id = str(user.id)
    
    if len(list(db.query(Searchlist.search_name))) > 0:
        searchnames = db.query(Searchlist.search_name, Searchlist.usr_search_name).filter_by(user_id = user_id).all()
        if len(searchnames) > 0:
            search_savedlist = []
            for item in searchnames:
                search_name = item[0]
                searchlist_id = list(db.query(Searchlist).filter_by(search_name=search_name))[0].id
                saved = db.query(UserRatings.name).filter_by(saved="1",searchlist_id=searchlist_id).all()
                saved = list(itertools.chain(*saved))
                search_savedlist.append((item[1],saved))
                
                #else:
                    #query = 'DROP TABLE '+search+'_'+user_id
                    #db.execute(query)
                    #db._model_changes={}
                    #db.commit()
            return render_template('bbnmsavednames.html', search_savedlist=search_savedlist)
        else:
            return render_template('bbnmsavednames.html', search_savedlist = None)
    else:
        return render_template('bbnmsavednames.html', search_savedlist = None)

@bbnm.route('/recommendations', methods = ['GET','POST'])
@login_required
def recommend():
    username = session['username']
    user = db.query(User).from_statement(text("SELECT * FROM users where username=:username")).params(username=username).one()
    user_id = str(user.id)
    if request.method == 'POST':
        recommendations = []
        for k,v in request.form.items():
            recommendations.append((k,v))
        filename = '/home/airos/flask-web/babyname/recommendations_response.csv'
        myfile = open(filename,'w')
        wrtr = csv.writer(myfile)
        wrtr.writerow(recommendations)
        myfile.close()
    else:
        if len(list(db.query(Searchlist.search_name))) > 0:
            searchnames = db.query(Searchlist.search_name, Searchlist.usr_search_name).filter_by(user_id = user_id).all()
            if len(searchnames) > 0:
                recommendlist = []
                for item in searchnames:
                    search_name = item[0]
                    searchlist_id = list(db.query(Searchlist).filter_by(search_name=search_name))[0].id
                    liked = db.query(UserRatings.name).filter_by(rating="1",searchlist_id=searchlist_id).all()
                    random.shuffle(liked)
                    if len(liked) >= 10:
                        liked_random = liked[:10]
                        liked_random = list(itertools.chain(*liked_random))
                    else:
                        liked_random = liked[:len(liked)]
                        liked_random = list(itertools.chain(*liked_random))
                    rec5_list = []
                    for name in liked_random:
                        rec5_list.append(rec_5names(name))
                    
                    rec_list = list(itertools.chain(*rec5_list))
                    random.shuffle(rec_list)
                    recommendlist.append((item[1], rec_list))
                return render_template('bbnmrecommend.html', recommendlist=recommendlist)
            else:
                return render_template('bbnmrecommend.html', recommendlist = None)
        else:
            return render_template('bbnmrecommend.html', recommendlist = None)


@bbnm.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == "__main__":
    bbnm.run(host='0.0.0.0',debug=True)



