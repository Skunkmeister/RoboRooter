import React, { useEffect, useState } from 'react';

const DepthDisplay = ({ depthData }) => {
    const [imageSrc, setImageSrc] = useState(null);

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
            case '16uc1':
                mimeType = 'image/png'; // or 'image/jpeg' depending on your data format
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

    useEffect(() => {
        // Update the image source every 100 milliseconds (adjust as needed)
        const interval = setInterval(() => {
            if (depthData) {
                setImageSrc(createDataUrl(depthData));
            }
        }, 100);

        // Clean up the interval on component unmount
        return () => clearInterval(interval);
    }, [depthData]);

    return (
        <div>
            <div>
                <h3>Depth Camera</h3>
                {imageSrc ? (
                    <img src={imageSrc} alt="Depth Image" width="400" height="600" />
                ) : (
                    <p>No Depth data available</p>
                )}
            </div>
        </div>
    );
};

export default DepthDisplay;
