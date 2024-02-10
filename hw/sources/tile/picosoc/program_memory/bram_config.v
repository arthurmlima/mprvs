
`timescale 1 ns / 1 ps
	module bram_config #
	(
		// Users to add parameters here

        parameter         S_NUM_COL             =   4,
        parameter         S_COL_WIDTH           =   8,
        parameter         S_ADDR_WIDTH          =  16, 
        parameter         S_DATA_WIDTH         =  S_NUM_COL*S_COL_WIDTH,  // Data  Width in bits
		parameter integer C_S00_AXI_DATA_WIDTH	= 32,
		parameter integer C_S00_AXI_ADDR_WIDTH	= 4
	)
	(

     input clkB,
     input enaB,
     input [S_NUM_COL-1:0] weB,
     input [S_ADDR_WIDTH-1:0] addrB,
     input [S_DATA_WIDTH-1:0] dinB,
     output reg [S_DATA_WIDTH-1:0] doutB,
     output reset_riscv,
     
		// User ports ends
		// Do not modify the ports beyond this line


		// Ports of Axi Slave Bus Interface S00_AXI
		input wire  s00_axi_aclk,
		input wire  s00_axi_aresetn,
		input wire [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_awaddr,
		input wire [2 : 0] s00_axi_awprot,
		input wire  s00_axi_awvalid,
		output wire  s00_axi_awready,
		input wire [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_wdata,
		input wire [(C_S00_AXI_DATA_WIDTH/8)-1 : 0] s00_axi_wstrb,
		input wire  s00_axi_wvalid,
		output wire  s00_axi_wready,
		output wire [1 : 0] s00_axi_bresp,
		output wire  s00_axi_bvalid,
		input wire  s00_axi_bready,
		input wire [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_araddr,
		input wire [2 : 0] s00_axi_arprot,
		input wire  s00_axi_arvalid,
		output wire  s00_axi_arready,
		output wire [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_rdata,
		output wire [1 : 0] s00_axi_rresp,
		output wire  s00_axi_rvalid,
		input wire  s00_axi_rready
	);
    wire	[31:0] s_doutB;
// Instantiation of Axi Bus Interface S00_AXI
	s_bramconfig # ( 
		.C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH),
		.C_S_NUM_COL   (S_NUM_COL   ),
		.C_S_COL_WIDTH (S_COL_WIDTH ),
		.C_S_ADDR_WIDTH(S_ADDR_WIDTH)
			
		
	) s_bramconfig_inst (

	  // portas para o acesso pelo risc
	 .clkB(clkB),
     .enaB (enaB),
     .weB(weB),
     .addrB(addrB),
     .dinB(dinB),
     .doutB(s_doutB),
     .reset_riscv(reset_riscv),

      // fim portas para o acesso pelo risc 
	   
	
	
	
		.S_AXI_ACLK(s00_axi_aclk),
		.S_AXI_ARESETN(s00_axi_aresetn),
		.S_AXI_AWADDR(s00_axi_awaddr),
		.S_AXI_AWPROT(s00_axi_awprot),
		.S_AXI_AWVALID(s00_axi_awvalid),
		.S_AXI_AWREADY(s00_axi_awready),
		.S_AXI_WDATA(s00_axi_wdata),
		.S_AXI_WSTRB(s00_axi_wstrb),
		.S_AXI_WVALID(s00_axi_wvalid),
		.S_AXI_WREADY(s00_axi_wready),
		.S_AXI_BRESP(s00_axi_bresp),
		.S_AXI_BVALID(s00_axi_bvalid),
		.S_AXI_BREADY(s00_axi_bready),
		.S_AXI_ARADDR(s00_axi_araddr),
		.S_AXI_ARPROT(s00_axi_arprot),
		.S_AXI_ARVALID(s00_axi_arvalid),
		.S_AXI_ARREADY(s00_axi_arready),
		.S_AXI_RDATA(s00_axi_rdata),
		.S_AXI_RRESP(s00_axi_rresp),
		.S_AXI_RVALID(s00_axi_rvalid),
		.S_AXI_RREADY(s00_axi_rready)
	);

	// Add user logic here
assign s_doutB=doutB;
	// User logic ends

	endmodule
