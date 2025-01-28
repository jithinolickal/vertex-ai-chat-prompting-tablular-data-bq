// src/components/ChatContent/ChatContent.jsx
import { Avatar, Tag } from "antd";
import "./ChatContent.css";
import { useEffect, useRef } from "react";
import {
  Sandpack,
  SandpackLayout,
  SandpackPreview,
  SandpackProvider,
} from "@codesandbox/sandpack-react";

const ChatContent = ({ messages }) => {
  console.log(messages);
  const messagesEndRef = useRef(); // Ref for enabling auto scroll down

  useEffect(() => {
    // Auto scroll down to latest message
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  return (
    <div className="chat-content">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`message-container ${
            message.type === "user" ? "user-message" : "ai-message"
          }`}
        >
          {message.type === "ai" && (
            <Avatar className="ai-avatar" alt="AI">
              AI
            </Avatar>
          )}
          <div className="message-content">
            {message.response_type === "narrative" && (
              <>
                {message.content}
                {message.sql && (
                  <details style={{ overflow: "scroll" }}>
                    <summary>SQL</summary>

                    <pre>{message.sql}</pre>
                  </details>
                )}
              </>
            )}
            {message.type === "user" && message.content}
            {message.type === "ai" && message.response_type === "chart" && (
              <div>
                <h4>Chart</h4>
                <SandpackProvider
                  template="react"
                  customSetup={{
                    dependencies: {
                      recharts: "latest",
                    },
                  }}
                  files={{
                    "App.js": message.content,
                    //                   "App.js": `import React from 'react';
                    // import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

                    // const data = [
                    //   {
                    //     name: 'Unassigned',
                    //     cost: 2988923.13
                    //   },
                    //   {
                    //     name: 'Microsoft Defender for Servers - Standard P2 Node',
                    //     cost: 1842490.74
                    //   },
                    //   {
                    //     name: 'Backup - GRS Data Stored - DE West Central',
                    //     cost: 1130403.65
                    //   }
                    // ];

                    // const Chart = () => {
                    //   return (
                    //   <ResponsiveContainer width="100%" height={300}>
                    //     <BarChart data={data}>
                    //       <CartesianGrid strokeDasharray="3 3" />
                    //       <XAxis
                    //         dataKey="name"
                    //         angle={-45}
                    //         textAnchor="end"
                    //         height={100}
                    //         interval={0}
                    //       />
                    //       <YAxis />
                    //       <Legend />
                    //       <Bar dataKey="cost" fill="#8884d8" name="Total Cost ($)" />
                    //     </BarChart>
                    //     </ResponsiveContainer>
                    //   );
                    // };

                    // export default Chart;`,
                  }}
                >
                  <SandpackLayout>
                    <SandpackPreview
                      showOpenInCodeSandbox={false}
                      showRefreshButton={true}
                    />
                  </SandpackLayout>
                </SandpackProvider>
              </div>
            )}
          </div>
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatContent;
