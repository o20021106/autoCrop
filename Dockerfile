FROM            python:3.6-onbuild
LABEL maintainer="o20021106@gmail.com"

COPY    entrypoint.sh /
RUN     chmod +x /entrypoint.sh 

ENV PATH="/auto_croppers/binary:${PATH}"

ENTRYPOINT      ["/entrypoint.sh"]
