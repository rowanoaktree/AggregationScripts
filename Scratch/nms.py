#Suzanne's code for getting rid of overlapping boxes
def overlap ( predicted_bb, output ):
    '''
    Output = a list of kept bounding boxes
    Takes input predicted_bb and list of kept bounding box predictions.
    Returns True if the predicted bounding box has significant overlap with 
    an existing bounding box prediction in the output list.
    Returns False otherwise
    '''
    if ( WEAKENED ):
        return False
    
    for box in output:
        existing_box = box[0:4] # only look at the existing box, not the score
        
        iou = compute_iou ( predicted_bb, existing_box )
        if ( iou >= kOverlapThreshold ):
            return True
    # if we've gotten to the end of the output files and not returned True, 
    # there is no overlapping box
    return False

def predict_boxes(heatmap, match_filter):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []
    box_height, box_width, _ = match_filter.shape

    # Find where the heatmap score is greater than or equal to the threshold
    match_indices = np.where ( heatmap >= kThreshold )
    i_coord = match_indices[0]
    j_coord = match_indices[1]
    if ( len ( i_coord ) == 0 or len ( j_coord ) == 0 ):
        # return empty array if none of the heatmap score is above the threshold
        return output;
    
    for i,j in zip(match_indices[0], match_indices[1]):
        score = heatmap[i][j]
        predicted_bb = [int(j), int(i), int(j+box_width), int(i+box_height)]
        '''
        LOOK HERE
        '''
        if ( not overlap ( predicted_bb, output ) ):
            output.append ( [ int(j), int(i), int(j+box_width), int(i+box_height), score] )

    '''
    END YOUR CODE
    '''

    return output