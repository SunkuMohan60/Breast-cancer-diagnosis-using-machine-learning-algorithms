#!/usr/bin/env python
import os
# top of the file
import sys, asyncio

# if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BreastCancerDiagnosis.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
