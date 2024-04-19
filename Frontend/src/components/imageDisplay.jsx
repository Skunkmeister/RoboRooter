import React from 'react';

const ImageDisplay = ({ imageData }) => {
    return (
        <div>
            {imageData && (
                <div>
                    <h3>RGB Camera</h3>
                    <img src={`data:image/jpeg;base64,${imageData.data}`} alt="ROS Image" />
                </div>
            )}
        </div>
    );
}

export default ImageDisplay;
