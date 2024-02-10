-- file: pixel_mem.vhd
-- description: pixel array per processPE_ing element
-- pixel PE_indexPE_ing - x coordPE_inate: column
--		  - y coordPE_inate: row
--                - PE_intermediary image PE_index: step
--                - current frame: frame

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity PE_dec is

generic(
    x_init          : integer := 0;
    y_init          : integer := 0;
    img_width       : natural;
    img_height      : natural;
    n_frames        : natural;
    n_steps         : natural;
    pix_depth       : natural;
    subimg_width    : natural;
    subimg_height   : natural;
    buffer_length   : natural
);

port(

	clk                         : in std_logic;
	rst                         : in std_logic;
	PE_ready_DEC_EB             : inout std_logic := '1'; --sinal indica se o decoder est? pronto para receber a mensagem 
	PE_stall_out_DEC            : inout std_logic := '0';
	PE_stall_in_DEC             : in std_logic ; 
    PE_in_x_dest                : in std_logic_vector(img_width-1 downto 0);
    PE_in_y_dest                : in std_logic_vector(img_height-1 downto 0);
    PE_in_x_orig                : in std_logic_vector(img_width-1 downto 0);
    PE_in_y_orig                : in std_logic_vector(img_height-1 downto 0);
    PE_in_fb                    : in std_logic;
    PE_out_send                 : out std_logic := '0';
    PE_out_direction            : out std_logic_vector(5 downto 0);
    PE_valid_EB_DEC             : in      std_logic;
    PE_out_new_EB_DEC           : in std_logic
    
);
end entity PE_dec;

architecture behavioral of PE_dec is

    --signal PE_out_direction : std_logic_vector(5 downto 0);--hotspot: PixMem, PE, N, S, E, W
    
    --signal pixel : natural := 0;
    signal x_dest: natural ;
    signal y_dest: natural ;
    signal x_dest_orig: natural ;
    signal y_dest_orig: natural ;
    --signal step  : natural := 0;
    --signal frame : natural := 0;
    signal x_orig: natural;
    signal y_orig: natural;
    --signal fb    : std_logic := '0';
    
    constant x_end : natural := x_init + subimg_width-1; -- end of region
    constant y_end : natural := y_init + subimg_height-1; -- end of region
    
    signal x_diff : integer;-- := x_dest-x_PE_init; -- distance to destPE_iny
    signal y_diff : integer;-- := y_dest-y_PE_init; -- distance to destPE_iny
    
    signal NW : std_logic := '0';
    signal SW : std_logic := '0';
    signal NE : std_logic := '0';
    signal SE : std_logic := '0';
    
    signal counter : natural := 0;
    signal state : std_logic := '0';
begin
    
    x_dest <= to_integer(unsigned(PE_in_x_dest));
    y_dest <= to_integer(unsigned(PE_in_y_dest));
    x_orig <= to_integer(unsigned(PE_in_x_orig));
    y_orig <= to_integer(unsigned(PE_in_y_orig));
    x_diff <= x_dest-x_init;
    y_diff <= y_dest-y_init;    

process(clk,rst) is
begin
    if(rst='1')then
        PE_out_direction <= "000000";

        PE_out_send    <= '0';
   
    else
       if(rising_edge(clk))then
       
       if(PE_out_new_EB_DEC = '1')then --
       state<= '0';
 --
       end if; 
       
       case state is 
       
       when '0' =>
            if(PE_out_new_EB_DEC = '1')then --  
            state<='1';
            PE_stall_out_DEC <= '1';
            
                   if(PE_in_fb='0')then

                           if( (x_dest>=x_init)and(x_dest<=x_end)and(y_dest>=y_init)and(y_dest<=y_end) ) then
                                
                                    PE_out_direction <= "100000"; 

                                   PE_out_send    <= '1';
                                   PE_ready_DEC_EB <= '0';

                                   
                           elsif( (x_dest>=x_init)and(x_dest<=x_end) )then -- rPE_oute PE_in Y (North/SPE_outh)
                               -- the above condition says that the message is PE_in the same X range of the local tile
                               -- defPE_ine now if rPE_oute North or SPE_outh
                               if( y_dest<y_init )then -- rPE_oute North
                                   PE_out_direction <= "001000";

                                   PE_out_send    <= '1';  
                                   PE_ready_DEC_EB <= '0';                                    
                               else -- rPE_oute SPE_outh
                                    PE_out_direction <= "000100";    
        
                                    PE_out_send    <= '1';   
                                    PE_ready_DEC_EB <= '0';                                       
                               end if;
                           elsif( (y_dest>=y_init)and(y_dest<=y_end) )then -- rPE_oute PE_in X (East/West)
                                -- the above condition says that the message is PE_in the same Y range of the local tile
                                -- defPE_ine now if rPE_oute East or West
                                if( x_dest<x_init)then -- rPE_oute West
                                    PE_out_direction <= "000001";   
      
                                    PE_out_send    <= '1';      
                                    PE_ready_DEC_EB <= '0';                                    
                                else -- rPE_oute East
                                    PE_out_direction <= "000010";       
          
                                    PE_out_send    <= '1';          
                                    PE_ready_DEC_EB <= '0';                                    
                                end if;
                           elsif( (x_dest<x_init)and(y_dest<y_init) )then -- North-West region
                                --defPE_ine now if rPE_oute North or West first
                                if(NW='0')then -- rPE_oute North
                                    PE_out_direction <= "001000";

                                    PE_out_send    <= '1';      
                                    PE_ready_DEC_EB <= '0';                           
                                else -- rPE_oute West
                                    PE_out_direction <= "000001";   
      
                                    PE_out_send    <= '1';      
                                    PE_ready_DEC_EB <= '0';                          
                                end if;
                                NW <= not(NW);
                           elsif( (x_dest<x_init)and(y_dest>y_end) )then -- SPE_outh-West region
                                -- defPE_ine now if rPE_oute SPE_outh or West first
                                if(SW='0')then -- rPE_oute SPE_outh
                                    PE_out_direction <= "000100";    
         
                                    PE_out_send    <= '1';      
                                    PE_ready_DEC_EB <= '0';                                 
                                else -- rPE_oute West
                                    PE_out_direction <= "000001";   
         
                                    PE_ready_DEC_EB <= '0';                       
                                end if;       
                                SW <= not(SW);                    
                           elsif( (x_dest>x_end)and(y_dest<y_init) )then -- North-East region
                                -- defPE_ine now if rPE_oute North or East first
                                if(NE='0')then -- rPE_oute North
                                    PE_out_direction <= "001000";

                                    PE_out_send    <= '1';                                 
                                    PE_ready_DEC_EB <= '0';
                                else -- rPE_oute East
                                    PE_out_direction <= "000010";       
          
                                    PE_out_send    <= '1';    
                                    PE_ready_DEC_EB <= '0';                              
                                end if;
                                NE <= not(NE);
                           elsif( (x_dest>x_end)and(y_dest>y_end) )then -- SPE_outh-East region
                                -- defPE_ine now if rPE_oute SPE_outh or East first
                                if(SE='0')then -- rPE_oute SPE_outh
                                    PE_out_direction <= "000100";    
        
                                    PE_out_send    <= '1';              
                                    PE_ready_DEC_EB <= '0';                         
                                else  -- rPE_oute East
                                    PE_out_direction <= "000010";       
           
                                    PE_out_send    <= '1';          
                                    PE_ready_DEC_EB <= '0';                        
                                end if;
                                SE <= not(SE);
                           end if;
                           
                       else
                      
                       --------------- FB = 1 = backward message
                        --PE_out_send <= '0';

                          if( (x_orig>=x_init)and(x_orig<=x_end)and(y_orig>=y_init)and(y_orig<=y_end) )then
--            
                                   PE_out_direction <= "010000"; -- rPE_oute to ProcessPE_ing Element

                                   PE_out_send    <= '1';   
                                   PE_ready_DEC_EB <= '0';                         
                           
                                  
                          elsif( (x_orig>=x_init)and(x_orig<=x_end) )then
                              -- the above condition says that the message is PE_in the same X range of the local tile
                              -- defPE_ine now if rPE_oute North or SPE_outh
                              if( y_orig<y_init )then -- rPE_oute North
                                  PE_out_direction <= "001000";

                                  PE_out_send    <= '1';     
                                  PE_ready_DEC_EB <= '0';                               
                              else -- rPE_oute SPE_outh
                                   PE_out_direction <= "000100";    
         
                                   PE_out_send    <= '1';         
                                   PE_ready_DEC_EB <= '0';                                
                              end if;
                          elsif( (y_orig>=y_init)and(y_orig<=y_end) )then -- rPE_oute PE_in X (East/West)
                               -- the above condition says that the message is PE_in the same Y range of the local tile
                               -- defPE_ine now if rPE_oute East or West
                               if( x_orig<x_init)then -- rPE_oute West
                                   PE_out_direction <= "000001";   
        
                                   PE_out_send    <= '1';           
                                   PE_ready_DEC_EB <= '0';                               
                               else -- rPE_oute East
                                   PE_out_direction <= "000010";       
            
                                   PE_out_send    <= '1';             
                                   PE_ready_DEC_EB <= '0';                                 
                               end if;
                          elsif( (x_orig<x_init)and(y_orig<y_init) )then -- North-West region
                               --defPE_ine now if rPE_oute North or West first
                               if(NW='0')then -- rPE_oute North
                                   PE_out_direction <= "001000";

                                   PE_out_send    <= '1';         
                                   PE_ready_DEC_EB <= '0';                        
                               else -- rPE_oute West
                                   PE_out_direction <= "000001";   
       
                                   PE_out_send    <= '1';     
                                   PE_ready_DEC_EB <= '0';                          
                               end if;
                               NW <= not(NW);
                          elsif( (x_orig<x_init)and(y_orig>y_end) )then -- SPE_outh-West region
                               -- defPE_ine now if rPE_oute SPE_outh or West first
                               if(SW='0')then -- rPE_oute SPE_outh
                                   PE_out_direction <= "000100";    
         
                                   PE_out_send    <= '1';     
                                   PE_ready_DEC_EB <= '0';                                 
                               else -- rPE_oute West
                                   PE_out_direction <= "000001";   
        
                                   PE_out_send    <= '1';         
                                   PE_ready_DEC_EB <= '0';                       
                               end if;       
                               SW <= not(SW);                    
                          elsif( (x_orig>x_end)and(y_orig<y_init) )then -- North-East region
                               -- defPE_ine now if rPE_oute North or East first
                               if(NE='0')then -- rPE_oute North
                                   PE_out_direction <= "001000";

                                   PE_out_send    <= '1';                                 
                                   PE_ready_DEC_EB <= '0';
                               else -- rPE_oute East
                                   PE_out_direction <= "000010";       
            
                                   PE_out_send    <= '1';        
                                   PE_ready_DEC_EB <= '0';                        
                               end if;
                               NE <= not(NE);
                          elsif( (x_orig>x_end)and(y_orig>y_end) )then -- SPE_outh-East region
                               -- defPE_ine now if rPE_oute SPE_outh or East first
                               if(SE='0')then -- rPE_oute SPE_outh
                                   PE_out_direction <= "000100";    
        
                                   PE_out_send    <= '1';    
                                   PE_ready_DEC_EB <= '0';                                
                               else -- rPE_oute East
                                   PE_out_direction <= "000010";       
            
                                   PE_out_send    <= '1';    
                                    PE_ready_DEC_EB <= '0';                              
                               end if;
                               SE <= not(SE);
                          end if;
                       
                       
                       end if;-- end fb
                   --end if;
                 --  counter <= 1;
                   
               end if;--PE_in ack=0 and PE_in_new=1
       when '1' =>

                if(PE_stall_in_DEC = '1')then
                
                    PE_ready_DEC_EB <= '1'; -- indica para o EB  que o decoder est? livre para receber msg
                    state<='0';
                    PE_out_send <= '0';
                    PE_stall_out_DEC <= '0';                   
                end if;
                
           when others =>
           
           end case;
        end if;
            
    end if;
end process;


end architecture behavioral;
