import React from 'react';

const DepthDisplay = ({ depthData }) => {


    return (
        <div>
            <div>
                <h3>Depth Camera</h3>
                {imageSrc ? (
                    <img src={"data:image/jpg;base64," + depthData.data} alt="Depth Image"  />
                ) : (
                    <p>No Depth data available</p>
                )}
            </div>
        </div>
    );
};

export default DepthDisplay;
