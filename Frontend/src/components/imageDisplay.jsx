import React from 'react';

const ImageDisplay = ({ imageData }) => {
    // Function to create a data URL from the raw image data
    const createDataUrl = (imageData) => {
        const { width, height, encoding, is_bigendian, step, data } = imageData;

        // Determine the MIME type based on the image encoding
        let mimeType;
        switch (encoding) {
            case 'rgb8':
                mimeType = 'image/jpeg';
                break;
            case 'rgba8':
                mimeType = 'image/png';
                break;
            default:
                console.error('Unsupported image encoding:', encoding);
                return null;
        }

        const uint8Array = new Uint8Array(data);
        const blob = new Blob([uint8Array], { type: mimeType });
        return URL.createObjectURL(blob);
    };

    return (
        <div>
            <div>
                <h3>RGB Camera</h3>
                {imageData ? (
                    <img src={createDataUrl(imageData)} alt="ROS Image" />
                ) : (
                    <p>No Image data available</p>
                )}
            </div>
        </div>
    );
};

export default ImageDisplay;
