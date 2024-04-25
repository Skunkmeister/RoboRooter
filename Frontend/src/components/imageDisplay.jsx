import React from 'react';
import placeholderImg from './assets/img/placeholder.png'

const ImageDisplay = ({ imageData }) => {
    // Function to create a data URL from the raw image data

    return (
        <div>
            <div>
                <h3>RGB Camera</h3>
                {imageData ? (
                    <img src={"data:image/jpg;base64," + imageData.data} alt="ROS Image"  />
                ) : (
                    <img src={placeholderImg} alt="No data" />
                )}
            </div>
        </div>
    );
};

export default ImageDisplay;
