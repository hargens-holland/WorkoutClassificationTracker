`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/17/2026 12:07:05 PM
// Design Name: 
// Module Name: weight_bram_ctrl
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


module weight_bram_ctrl(
        input  logic       clk,
        input  logic       rst_n,
        input  logic       new_addr_valid,
        input  logic [1:0] layer_sel,       // 00: Layer 1, 01: Layer 2, 10: Output, 11: Bias 
        input  logic [6:0] neuron,          // Neuron we are calculating (128 is max neurons)
        input  logic [6:0] input_num,       // Max is 128 inputs 
        input  logic [7:0] bias_sel,        // [127:0] Layer 1 biases, [191:128] Layer 2 , [196:192] Output
        output logic       weight_data_valid,
        output logic [7:0] weight_data      // Data to MAC
    );

    // BRAM layout (single BRAM, addressed by layer + neuron + input):
    //   Layer 1: 34*128  = 4,352 weights  (addr 0x0000 - 0x10FF)
    //   Layer 2: 128*64  = 8,192 weights  (addr 0x1100 - 0x30FF)
    //   Output:  64*3    =   192 weights  (addr 0x3100 - 0x31BF)
    //   Biases:  128+64+3=   195 biases   (addr 0x31C0 - 0x3282)
    //   Total:             12,931 entries
    // all_weights.coe = 13056
    logic [13:0] address;
    logic [13:0] layer_base;
    logic [12:0] neuron_base;

    logic        new_addr_valid_flop;

    // Depending on the layer create a base number where the address will start
    assign layer_base = (layer_sel == 2'b00) ? 14'b00_0000_0000_0000                                                  // Layer 1
                                             : (layer_sel == 2'b01) ? 14'b01_0001_0000_0000                           // Layer 2
                                                                    : (layer_sel == 2'b10) ? 14'b11_0001_0000_0000    // Output
                                                                                           : 14'b11_0001_1100_0000;   // Bias

    // Depending on which neuron create another base number where the address will add to the layer base
    // each neuron has as many weights as there are inputs so also depends on which layer
    assign neuron_base = (layer_sel == 2'b00) ? (neuron * 34)                                               // 34 inputs 128 neurons
                                              : (layer_sel == 2'b01) ? (neuron * 128)                       // 128 inputs 64 neurons
                                                                     : (layer_sel == 2'b10) ? (neuron * 64) // 64 inputs 5 neurons
                                                                                            : 13'd0;        // When biasing there is not neurons

    // Calculate final adress with layer and neuron base and which input number we are on for all layers but the biasing layer
    // For the bias layer it is just the layer base along with the which bias we are getting
    assign address = (layer_sel == 2'b11) ? (layer_base + bias_sel) : (layer_base + neuron_base + input_num);

    blk_mem_gen_0 weights_and_biases (
      .clka(clk),           // input wire clka
      .ena(new_addr_valid), // input wire ena
      .addra(address),      // input wire [13 : 0] addra
      .douta(weight_data)   // output wire [7 : 0] douta
    );

    // BRAM takes two cycles to access data so flop the addr valid to detect when the weight_data is valid
    always_ff @(posedge clk) begin
        new_addr_valid_flop <= ~rst_n ? 1'b0 : new_addr_valid;
        weight_data_valid   <= ~rst_n ? 1'b0 : (new_addr_valid_flop & new_addr_valid);
    end
endmodule
