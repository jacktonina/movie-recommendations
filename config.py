import os


basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    memory = '10g'
    pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
    os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
