from flask import Flask
from celery import Celery

import logging.config

logging.config.fileConfig(fname='base_logger.conf', disable_existing_loggers=False)

def create_app():
    app = Flask(__name__)
    app.config.from_object('app_config')
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
    
    with app.app_context(): 
        print(__name__)
        from .api.cropping import cp
        app.register_blueprint(cp)  

    return app 


def create_celery_app(app=None):
    abstract = True

    app = app or create_app()
    celery = Celery(__name__, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    celery.app = app
    return celery

