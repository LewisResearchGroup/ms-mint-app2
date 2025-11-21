general
- [ ] cuando se eliminan elementos (toda la tabla?) la db no disminuye en tamaño, siempre crece. crear un boton en workspaces para "manual cleaning" haciendo copy 


explorer
- [ ] (*) añadir soporte para unidades que no sean el home
- [ ] catch errores de procesamiento de archivos ms_files para evitar crashes 
- [ ] (enhancement) probar el componente transfer pero sin eliminar elementos de la izquierda 
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
- [ ] (*) el archivo de targets que se genera no lo lee. revisar las columnas que son None o que se computan internamente

Optimization
- [x] Chromatogram View. Cuando se salva, el rslider se reinicia al modo completo.
- [x] La leyenda por grupos no permite ocultar un solo elemento, sin ocultar a todo el grupo
- [x] anadir un boton para cambiar entre groupclick de la leyenda.
- [x] RT-representation redondearlo para mejor aspecto visual
- [x] Cuando son muchos elementos para computar el cromatograma, duckdb no notifica bien el progreso. El modal no se cierra tampoco.
- [ ] añadida la opción para la seleccion del número de cpus y ram para la generación de los cromatogramas (revisar)
- [x] añadir seleccion para ms_type
- [x] falla el slider
- [x] con el filtro de ms_type se requiere modificar las sample_type que se muestran en el tree
- [x] cuando esta en log scale y se hace zoom se reinicia el eje
- [x] el log scale deja de funcionar
- [ ] sin metadata no hay ningun ms_file marcado para optimizacion y debe notificarse en "Compute Chromatograms" para no computar nada
- [ ] similar para targets
- [ ] barra de progreso. Ver si la query internamente puede mostrar progreso, de lo contrario subdividir en lotes para mostrar progreso y asi mejorar la UX 
- [ ] rollback delete cromatogramas
- [ ] remover tentativamente la linea del RT (se computara de forma automatica)
- [ ] order by tiene problemas. no se ordena toda la tabla antes de paginar

Processing
- [x] solo esta procesando los que estan marcados para optimizacion. falta computar los cromatogramas de todos los ms_files
- [ ] la busqueda tiene mala interacion con el numero de paginas
- [ ] remover results cuando se elimine samples y targets
- [ ]


Workspaces
- [x] description al modal de crear ws. (cpu y ram se implementaron individualmente para el procesamiento de archivos y generacion de chromatograms)

mint
- [ ] manage target processing errors



# Features
- [ ] extraer sample type desde el nombre del archivo. por ejemplo, si tiene MHPool, el sample type es "MHPool" (elementos blank, mhpool, std)
- [ ] colores por grupos. para grupos standard asignar colores a todo el grupo, para el resto, asignar colores individualmente
- [ ] tamaño de card plot que ocupe todo el layout (hardcoded)
- [ ] (*) el filtrado por mz se cambia a filtrado por scan_id, si mz no existe se asigna 0 (incluye todos los valores)
- [ ] buscar compuestos en optimization (barra de busqueda)
- [ ] incluir opcion para 50 cards en optimization
- [ ] establecer aspect ratio para numero de elementos definidos en la paginacion (4, 10, 20, 50)
- [ ] (*) ordenar por mz_mean por default en optimization (tambien en targets por default)
- [ ] preview solo el rt span
- [ ] view renderizar el grafico con zoom alrededor del rt span (que incluya un 20% del eje x)
- [ ] (*) acceder a los discos externos (usb, sd, etc)
- [ ] arreglar el save automatico cuando se cierra la modal de view
- [ ] cambiar el boton Ok/cancel del modal de confirmacion de cierre sin guardar por save/cancel

