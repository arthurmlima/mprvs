module ram
 #(
//--------------------------------------------------------------------------

parameter   NUM_COL             =   4,
parameter   COL_WIDTH           =   8,
parameter   ADDR_WIDTH          =  16, 

// Addr  Width in bits : 2 *ADDR_WIDTH = RAM Depth
parameter   DATA_WIDTH      =  NUM_COL*COL_WIDTH  // Data  Width in bits
    //----------------------------------------------------------------------

  ) (
     input clkA,
     input enaA, 
     input [NUM_COL-1:0] weA,
     input [ADDR_WIDTH-1:0] addrA,
     input [DATA_WIDTH-1:0] dinA,
     output reg [DATA_WIDTH-1:0] doutA,
     output reg doa_ok, 

     
     input clkB,
     input enaB,
     input [NUM_COL-1:0] weB,
     input [ADDR_WIDTH-1:0] addrB,
     input [DATA_WIDTH-1:0] dinB,
     output reg [DATA_WIDTH-1:0] doutB
     );
   // Core Memory  
   reg [DATA_WIDTH-1:0]   ram_block [(6500)-1:0];
   integer                i;
   
       initial begin
        
        ram_block[0]<= 32'h 00018537;
        ram_block[1]<= 32'h 00050513;
        ram_block[2]<= 32'h 00f00413;
        ram_block[3]<= 32'h 00852023;
        
        
       
        end
   // Port-A Operation
   always @ (posedge clkA) begin
      if(enaA) begin

         for(i=0;i<NUM_COL;i=i+1) begin
            if(weA[i]) begin
               ram_block[addrA][i*COL_WIDTH +: COL_WIDTH] <= dinA[i*COL_WIDTH +: COL_WIDTH];
            end

         end
         doutA <= ram_block[addrA]; 
         doa_ok<=1'b1;
         end
               else 
        doa_ok<=1'b0;    
   end
   // Port-B Operation:
   always @ (posedge clkB) begin
      if(enaB) begin
         for(i=0;i<NUM_COL;i=i+1) begin
            if(weB[i]) begin
               ram_block[addrB][i*COL_WIDTH +: COL_WIDTH] <= dinB[i*COL_WIDTH +: COL_WIDTH];
            end        
         end            
         doutB <= ram_block[addrB];  
      end   
   end
endmodule // bytewrite_tdp_ram_rf