# Generated by Django 5.1.5 on 2025-03-17 09:42

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0007_alter_userprofile_image'),
    ]

    operations = [
        migrations.CreateModel(
            name='RequestFileModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('requester', models.EmailField(max_length=254)),
                ('request_date', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(default='Pending', max_length=255)),
                ('file_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='app.uploadfilemodel')),
            ],
            options={
                'db_table': 'RequestFileModel',
            },
        ),
    ]
