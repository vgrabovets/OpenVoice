from setuptools import setup, find_packages


setup(name='MyShell-OpenVoice',
      version='0.0.0',
      description='Instant voice cloning by MyShell.',
      long_description=open('README.md').read().strip(),
      long_description_content_type='text/markdown',
      keywords=[
            'text-to-speech',
            'tts',
            'voice-clone',
            'zero-shot-tts'
      ],
      url='https://github.com/myshell-ai/OpenVoice',
      project_urls={
        'Documentation': 'https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md',
        'Changes': 'https://github.com/myshell-ai/OpenVoice/releases',
        'Code': 'https://github.com/myshell-ai/OpenVoice',
        'Issue tracker': 'https://github.com/myshell-ai/OpenVoice/issues',
      },
      author='MyShell',
      author_email='ethan@myshell.ai',
      license='MIT License',
      packages=find_packages(),

      python_requires='>=3.9',
      zip_safe=False
      )
