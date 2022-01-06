FROM jjanzic/docker-python3-opencv
WORKDIR /src/app
COPY requirements.txt .
RUN python -m pip install twine cython requests tqdm
RUN python -m pip install cibuildwheel
CMD ["python"]

