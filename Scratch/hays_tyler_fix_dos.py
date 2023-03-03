

annos = []
#We iterate over each label event one at a time.
for label_event in usfws:
    #skip if the user skipped
    if label_event["Skipped"]==False:
        #for each bird that the user labeled iterate over each label geom that the user created.
        for bird in label_event['Label']:
            for geom_obj in label_event['Label'][bird]:
                #add the points to a 2 dimensional array
                polycoords = [[point['x'],point['y']] for point in geom_obj['geometry']]
                #Convert to a numpy array
                polycoords=np.array(polycoords)
                #Create a traditional bounding box
                minbox=BoundingBox(polycoords)
                #Convert bbox to COCO bbox format
                bbox = [minbox.minx, minbox.miny, minbox.width, minbox.height]
                #Create annotation and add to annos array.
                annotation = {
                "annotation_ID": len(annos)+1,
                "bbox": list(bbox),
                "filename": label_event["External ID"],
                "labeler": label_event["Created By"],
                "category": bird
                }
                annos.append(annotation)
annos