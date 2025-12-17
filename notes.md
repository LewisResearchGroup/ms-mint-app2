general
- [ ] cuando se eliminan elementos (toda la tabla?) la db no disminuye en tamaño, siempre crece. crear un boton en workspaces para "manual cleaning" haciendo copy 


explorer
- [x] (*) añadir soporte para unidades que no sean el home
- [x] catch errores de procesamiento de archivos ms_files para evitar crashes 
- [x] (enhancement) probar el componente transfer pero sin eliminar elementos de la izquierda 
  - doble click para transferir o seleccionar archivos
  - shift + click para seleccion multiple continua
  - ctrl + click para seleccion multiple discontinua
  - arrastrar para seleccionar archivos

ms_files
- [x] barra de progreso desincronizada en el procesamiento de archivos.la barra termina, pero sigue procesando.
- [x] regenerate color se activa automaticamente sin confirmar el modal (relacionado con el de abajo)
- [x] falla en repoblar la data cuando se vuelve a ms_file (relacionado con el de abajo)
- [x] regenerate esta activando el clear all
- [x] añadida la opción para la seleccion del número de cpus. No se estableció una configuracion global porque el 
      procesamiento de archivos y el procesamiento de duckdb ocupan diferente cantidad de ram por hilo. duckdb parece 
      ser muy eficiente, puede usar todos los cpus, pero consumir entre 3-4 GB, mientras que el multiprocessing puede 
      tomar 500-600 MB (para ms1, ms2 ? ) por hilo lo que puede hacer que se agote la RAM muy rápido. 
- [x] delete selected fail ms_data instead of
- [ ] cuando cancel el procesamiento, debe actualizarse la tabla


Targets
- [x] order dont work
- [ ] cuando falla un target no notifica bien
- [x] (*) el archivo de targets que se genera no lo lee. revisar las columnas que son None o que se computan internamente
- [ ] preselect no se marca en la tabla cuando se refresca

Optimization
- [x] Chromatogram View. Cuando se salva, el rslider se reinicia al modo completo.
- [x] La leyenda por grupos no permite ocultar un solo elemento, sin ocultar a todo el grupo
- [x] anadir un boton para cambiar entre groupclick de la leyenda.
- [x] RT-representation redondearlo para mejor aspecto visual
- [x] Cuando son muchos elementos para computar el cromatograma, duckdb no notifica bien el progreso. El modal no se cierra tampoco.
- [x] añadida la opción para la seleccion del número de cpus y ram para la generación de los cromatogramas (revisar)
- [x] añadir seleccion para ms_type
- [x] falla el slider
- [x] con el filtro de ms_type se requiere modificar las sample_type que se muestran en el tree
- [x] cuando esta en log scale y se hace zoom se reinicia el eje
- [x] el log scale deja de funcionar
- [x] sin metadata no hay ningun ms_file marcado para optimizacion y debe notificarse en "Compute Chromatograms" para no computar nada
- [x] similar para targets
- [x] barra de progreso. Ver si la query internamente puede mostrar progreso, de lo contrario subdividir en lotes para mostrar progreso y asi mejorar la UX 
- [x] rollback delete cromatogramas
- [x] remover tentativamente la linea del RT (se computara de forma automatica)
- [x] order by tiene problemas. no se ordena toda la tabla antes de paginar
- [x] cards no alineadas


Processing
- [x] solo esta procesando los que estan marcados para optimizacion. falta computar los cromatogramas de todos los ms_files
- [x] la busqueda tiene mala interacion con el numero de paginas
- [x] remover results cuando se elimine samples y targets


Workspaces
- [x] description al modal de crear ws. (cpu y ram se implementaron individualmente para el procesamiento de archivos y generacion de chromatograms)

mint
- [ ] manage target processing errors



# Features
- [ ] extraer sample type desde el nombre del archivo. por ejemplo, si tiene MHPool, el sample type es "MHPool" (elementos blank, mhpool, std)
- [x] colores por grupos. para grupos standard asignar colores a todo el grupo, para el resto, asignar colores individualmente
- [x] establecer aspect ratio para numero de elementos definidos en la paginacion (4, 10, 20, 50)
- [x] (*) el filtrado por mz se cambia a filtrado por scan_id, si mz no existe se asigna 0 (incluye todos los valores)
- [x] buscar compuestos en optimization (barra de busqueda)
- [x] incluir opcion para 50 cards en optimization
- [x] (*) ordenar por mz_mean por default en optimization (tambien en targets por default)
- [x] preview solo el rt span
- [x] view renderizar el grafico con zoom alrededor del rt span (que incluya un 20% del eje x)
- [x] (*) acceder a los discos externos (usb, sd, etc)
- [x] arreglar el save automatico cuando se cierra la modal de view
- [ ] cambiar el boton Ok/cancel del modal de confirmacion de cierre sin guardar por save/cancel
- [x] establecer number de records como opcion en el page size de la paginacion
- [x] actualizar la preview cuando se edite el rt span
- [x] En la preview quieren que se vea el rt span usando la escala de esa región
- [x] arreglar ticks en el eje y en log scale
- [x] arreglar Legend behavior. guarda el estado anterior. hay que reinicializarlo cuando se cierra la modal de view
- [x] en los chromatograms poner intensity = 1 en vez de 0, para evitar que se rompa el grafico en log scale
- [x] arreglar targets si tiene rt en min convertirlo a segundos 
- [x] arreglar el slider de rt span con el zoom
- [x] hacer pruebas de rendimiento de descarga de resultados. probar alternativas a duckdb con pandas o polars
- [x] cambiar colors por sample_type y paletas degradadas para cada grupo.
- [x] log file (parcialmente)
- [x] (*) corregir el catch de los errors en el procesamiento de archivos
- [x] cuando se borran los ultimos archivos de ms_files se va complemente la paginacion
- [x] poner ms_file_label y label del mismo ancho
- [ ] modal para notificar cuando borras tablas grandes
- [x] notificacion de duplicados hay que reducirlo
- [x] bajar el recurso de duckdb al 50%
- [ ] revisar el patron de tiempo de generacion de cromatogramas, hay un patron de poco tiempo, mucho tiempo
- [ ] arreglar el cancel de la generacion de cromatogramas porque no termina (asumo se por el while)
- [ ] en resultados cuando se filtra tiene mala interaccion con la paginacion. hay que moverla para la pagina 1. 
- [ ] Tambien debe ajustarse el numero de elementos de acuerdo al filtrado
- [x] aplicar el calculo por batches en processing
- [x] (*) Implementar los analisis
- [ ] establecer limites en y para el valor min > 1
- [x] probar el threshold
- [ ] z-order para las traces
- [ ] cambiar las tablas a una arquitectura basada en uuid4 para poder editar los nombre de los targets y otras cosas
- [ ] anotar nuevo target duplicando el cromatograma. se añade un nuevo target con otro rt-span usando el chromatograma precomputado 
- [ ] remover target desde la modal
- [x] probar selection box para el rt-span
- [ ] poner indicador de ocupado para los botones de descarga
- [x] los archivos descargados deben tener el identificador del ws y demas cosas
- [x] revisar cuando los targets se eliminan que no quede el bookmark marcado
- [ ] (*) hacer que cuando se levante la app mate cualquier instancia que exista de esta o procesos huerfanos
- [x] (*) establecer limites en y autorange cuando se hace el zoom a la region en la view
- [x] revisar cuando se sube una segunda tabla de targets si falla
- [x] mostrar progreso de procesamiento de archivos ms_file, metadata y targets
- [x] se muestra el mismo target en dos paginas continuas
- [x] eliminar un boton de generar color en ms files
- [x] add el boton para descargar metadata
- [ ] cambiar el Run order y demas columnas a group 1 - 5