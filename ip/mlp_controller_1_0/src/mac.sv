`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/12/2026 04:48:48 PM
// Design Name: 
// Module Name: mac_unit
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module mac(
        input  logic               clk,
        input  logic               valid_mac_inputs,
        input  logic signed [7:0]  point,        // move net point 
        input  logic signed [7:0]  weight,       // Weight from BRAM
        input  logic signed [19:0] acc_in,       // Previous accumulation sum
        output logic signed [20:0] acc_out,      // Updated sum
        output logic               valid_mac_out
    );

    logic [7:0] valid_point;
    logic [7:0] valid_weight;

    // valid signal flopped 4 times
    // if they are all 0 then the mac output is valid
    logic       valid_mac_inputs_1;
    logic       valid_mac_inputs_2;
    logic       valid_mac_inputs_3;
    logic       valid_mac_inputs_4;

    assign valid_point  = valid_mac_inputs ? point : 8'd0;
    assign valid_weight = valid_mac_inputs ? weight : 8'd0;


    always_ff @(posedge clk) begin
        valid_mac_inputs_1 <= valid_mac_inputs;
        valid_mac_inputs_2 <= valid_mac_inputs_1;
        valid_mac_inputs_3 <= valid_mac_inputs_2;
        //valid_mac_inputs_4 <= valid_mac_inputs_3;
    end

    assign valid_mac_out = ~(valid_mac_inputs_1 | valid_mac_inputs_2 | valid_mac_inputs_3);

    dsp_macro_0 mac (
      .CLK(clk),        // input wire CLK
      .A(valid_point),  // input wire [7 : 0] A
      .B(valid_weight), // input wire [7 : 0] B
      .C(acc_in),       // input wire [19 : 0] C
      .P(acc_out)       // output wire [20 : 0] P
    );

endmodule
