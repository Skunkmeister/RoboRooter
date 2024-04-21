import React, { Component } from 'react';
import Alert from "react-bootstrap/Alert"
import Config from "../data/config"
import ROSLIB from 'roslib';
import Navigation from "./navigation";
import ImageDisplay from "./imageDisplay"
import DepthDisplay from "./depthDisplay";
import CostmapDisplay from "./costmapDisplay";
import { Row, Col } from "react-bootstrap";

class Connection extends Component {
    state = {
        ros: new ROSLIB.Ros(),
        connected: false,
        imageData: null,
        depthData: null,
        costmapData: null
    }

    initConnection = () => {
        const rosbridgeServerUrl = `ws://${Config.ROSBRIDGE_SERVER_IP}:${Config.ROSBRIDGE_SERVER_PORT}`;
        this.state.ros.connect(rosbridgeServerUrl);

        this.state.ros.on("error", (error) => {
        console.error("WebSocket error:", error);
        });

        this.subscribeToTopic('/camera/rgb/image_raw', 'sensor_msgs/Image', 'imageData');
        this.subscribeToTopic('/camera/depth/image_raw', 'sensor_msgs/Image', 'depthData');
        this.subscribeToTopic('move_base/local_costmap/costmap', 'nav_msgs/OccupancyGrid', 'costmapData');

        // Event listeners for ROS connection status
        this.state.ros.on("connection", () => {
            console.log("Connection established successfully");
            this.setState({ connected: true });
        });

        this.state.ros.on("close", () => {
            console.log("Connection closed");
            this.setState({ connected: false });
        });
    }

    subscribeToTopic = (topicName, messageType, stateKey) => {
        const listener = new ROSLIB.Topic({
            ros: this.state.ros,
            name: topicName,
            messageType: messageType
        });

        listener.subscribe((message) => {
            console.log(`Received message on topic ${topicName}:`, message);
            this.setState({ [stateKey]: message });
        });
    }

    componentDidMount() {
        this.initConnection();
    }

    render() {

        const {connected, imageData, depthData, costmapData} = this.state;

        return (
            <div>
                <Row className="nomargin  nopadding">
                    <Col className="nomargin nopadding box-border">
                         <div>
                <Alert className="text-center m-3 nomargin nopadding  box-border" variant={connected ? "success" : "danger"}>
                    <h3> {connected ? "Online" : "Offline"} </h3>
                </Alert>
            </div>
                    </Col>
                </Row>
                <Row  className="nomargin nopadding">
                    <Col className="nomargin nopadding box-border">
                        <ImageDisplay imageData={imageData} className="box-border" />
                        <Navigation className="nomargin box-border" />

                    </Col>
                    <Col   className="nomargin nopadding box-border ">
                        <DepthDisplay depthData={depthData} />
                        <CostmapDisplay costmapData={costmapData}/>

                    </Col>
                </Row>
                
            </div>

        );
    }
}

export default Connection;
