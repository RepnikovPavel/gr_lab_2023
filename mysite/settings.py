import os

BASE_DIR = './'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'frontend/templates'),
                 os.path.join(BASE_DIR, 'my_app', 'templates', 'my_app')
                ]
    }
]