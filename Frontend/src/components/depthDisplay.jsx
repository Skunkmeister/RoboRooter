import React from 'react';

const DepthDisplay = ({ depthData }) => {
    return (
        <div>
            {depthData && (
                <div>
                    <h3>Depth Camera</h3>
                    {/* Assuming depthData is a base64 encoded image */}
                    <img src={`data:image/png;base64,${depthData}`} alt="Depth Image" />
                </div>
            )}
        </div>
    );
}

export default DepthDisplay;

