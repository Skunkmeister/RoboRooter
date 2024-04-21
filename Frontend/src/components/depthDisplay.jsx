import React from 'react';

const DepthDisplay = ({ depthData }) => {
    // Function to create a data URL from the raw depth data
    const createDataUrl = (depthData) => {
        const { width, height, encoding, is_bigendian, step, data } = depthData;

        // Determine the MIME type based on the depth data encoding
        let mimeType;
        switch (encoding) {
            case 'rgb8':
                mimeType = 'image/jpeg';
                break;
            case 'rgba8':
                mimeType = 'image/png';
                break;
            default:
                console.error('Unsupported depth data encoding:', encoding);
                return null;
        }

        // Construct a Uint8Array from the raw depth data
        const uint8Array = new Uint8Array(data);

        // Construct a Blob from the Uint8Array
        const blob = new Blob([uint8Array], { type: mimeType });

        // Construct a data URL from the Blob
        return URL.createObjectURL(blob);
    };

    return (
        <div>
            <div>
                <h3>Depth Camera</h3>
                {depthData ? (
                    <img src={createDataUrl(depthData)} alt="Depth Image" />
                ) : (
                    <p>No Depth data available</p>
                )}
            </div>
        </div>
    );
};

export default DepthDisplay;
