# Generated by Django 3.1.4 on 2022-03-21 09:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('records', '0002_auto_20201225_1227'),
    ]

    operations = [
        migrations.AddField(
            model_name='records',
            name='avatar',
            field=models.ImageField(blank=True, null=True, upload_to='users/'),
        ),
    ]
