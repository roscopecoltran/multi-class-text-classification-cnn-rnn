#!/bin/sh

case "$1" in

  'jupyter')
  	exec jupyter notebook --allow-root
	;;

  'tf-server')
  	exec python3 server.py
	;;

  'deepdetect')
  	exec dede -host 0.0.0.0
	;;

  'bash')
  	exec /bin/bash $@
  	;;

  *)
  	exec /bin/bash $@
	;;
esac